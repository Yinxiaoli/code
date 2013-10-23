#include "jsonrpc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifdef WIN32
#include <Winsock2.h>
#include <WS2tcpip.h>
#include <io.h>
#else
#include <unistd.h>
#include <netdb.h> 
#endif


JsonRpcServer::JsonRpcServer(int port) {
  functions = NULL;
  num_functions = 0;
  this->port = port;
  maxClients = omp_get_num_procs();
}

JsonRpcServer::~JsonRpcServer() {
  for(int i = 0; i < num_functions; i++)
    delete functions[i];
  free(functions);
}

JsonRpcMethodAbstract *JsonRpcServer::FindMethod(const char *name) {
  int ind = FindMethodInd(name);
  return ind >= 0 ? functions[ind] : NULL;
}

int JsonRpcServer::FindMethodInd(const char *name) {
  for(int i = 0; i < num_functions; i++)
    if(!strcmp(functions[i]->Name(), name))
      return i;
  return -1;
}

void JsonRpcServer::RegisterMethod(JsonRpcMethodAbstract *func) {
  if(!FindMethod(func->Name())) {
    functions = (JsonRpcMethodAbstract**)realloc(functions, sizeof(JsonRpcMethodAbstract*)*(num_functions+1));
    functions[num_functions++] = func;
  }
}

void JsonRpcServer::UnregisterMethod(const char *name) {
  int ind = FindMethodInd(name);
  if(ind >= 0) {
    for(int i = ind; i < num_functions-1; i++)
      functions[i] = functions[i+1];
    delete functions[ind];
    num_functions--;
  }
}

void JsonRpcServer::Shutdown() {
  shutdown = true;
  Json::Value req;
  req["method"] = "terminate";
  JsonRpcClientRequest("127.0.0.1", port, req);
}

void JsonRpcServer::RunServer() {
#ifdef WIN32
  {
    WSADATA WsaData;
    WSAStartup (0x0101, &WsaData);
  }
#endif

  shutdown = false;

  struct sockaddr_in serv_addr, cli_addr;
  SOCKET sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    fprintf(stderr, "ERROR opening socket\n");
    return;
  }
  
  memset((char *) &serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(port);
  if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
    fprintf(stderr, "ERROR binding socket\n");
    return;
  }


  fprintf(stderr, "Server listening on port %d\n", port);

  listen(sockfd,5);
  socklen_t clilen = sizeof(cli_addr);
  omp_lock_t lock;
  omp_init_lock(&lock);


  #pragma omp parallel num_threads(maxClients)
  {
    //#pragma omp single nowait
    {
      char *buf = new char[2000000];
      while(!shutdown) {
	omp_set_lock(&lock);
        SOCKET newsockfd = accept(sockfd, (struct sockaddr *)&cli_addr, &clilen);
	omp_unset_lock(&lock);
        if(shutdown) break;
        if (newsockfd < 0) { 
          fprintf(stderr, "ERROR accepting socket connection\n");
          break;
        }

        //#pragma omp task untied
        {
#ifdef WIN32
          int n = recv(newsockfd, buf, 1999999, 0);
#else
          int n = read(newsockfd, buf, 1999999);
#endif
          buf[n] = 0;

          ProcessRequest(newsockfd, buf);
//#pragma omp taskwait

#ifdef WIN32
          closesocket(newsockfd);
#else
          close(newsockfd);
#endif
        }
      }
    }
  }
#ifdef WIN32
  closesocket(sockfd);
  WSACleanup( );
#else
  close(sockfd);
#endif
}
  

void JsonRpcServer::ReturnJsonResponse(SOCKET  newsockfd, const Json::Value &response) {
  Json::FastWriter writer;
  std::string r = writer.write(response);
  char *str = (char*)malloc(strlen(r.c_str())+1);
  
  strcpy(str, r.c_str());

  //fprintf(stderr, "Server: %s\n", str);

#ifdef WIN32
  send(newsockfd, str, strlen(str), 0);
#else
  bool b = write(newsockfd, str, strlen(str));
#endif
  free(str);
}
  

Json::Value JsonRpcServer::Usage() {
  Json::Value retval;
  for(int i = 0; i < num_functions; i++) {
    Json::Value u = functions[i]->Usage();
    if(u != Json::Value::null) 
      retval[functions[i]->Name()] = u;
  }
  return retval;
}

void JsonRpcServer::PrintUsage() {
  Json::Value usage = Usage();
  Json::StyledWriter writer;
  fprintf(stderr, "%s\n", writer.write(usage).c_str());
}



Json::Value JsonRpcClientRequest(const char *hostname, int port, Json::Value &req, bool debug) {
#ifdef WIN32
  {
    WSADATA WsaData;
    WSAStartup (0x0101, &WsaData);
  }
#endif
  
  req["jsonrpc"] = "2.0";
  req["id"] = "0";


  sockaddr_in sin;
  SOCKET sock = socket (AF_INET, SOCK_STREAM, 0);
  if (sock == -1) {
    fprintf(stderr, "Failed to create socket\n");
    return Json::Value::null;
  }
  sin.sin_family = AF_INET;
  sin.sin_port = htons( (unsigned short)port);

  struct hostent *host_addr = gethostbyname(hostname);
  if(host_addr==NULL) {
#ifdef WIN32
    closesocket(sock);
#else
    close(sock);
#endif
    fprintf(stderr, "Failed to find hostname %s\n", hostname);
    return Json::Value::null;
  }
  sin.sin_addr.s_addr = *((int*)*host_addr->h_addr_list) ;

  if( connect (sock,(const struct sockaddr *)&sin, sizeof(sockaddr_in) ) == -1 ) {
#ifdef WIN32
    closesocket(sock);
#else
    close(sock);
#endif
    fprintf(stderr, "Failed to find connect to socket\n");
    return Json::Value::null;
  }

  Json::FastWriter writer;
  std::string r = writer.write(req);
  char *str = (char*)malloc(strlen(r.c_str())+1);
  strcpy(str, r.c_str());
  if(debug) fprintf(stderr, "Client: %s\n", str);
  send(sock,str,strlen(str),0);
  int n, len = 0;
  do {
    str = (char*)realloc(str, len + 100001);
    n = recv(sock,str+len,100000,0);
    if(n > 0) len += n;
  } while(n > 0);
  str[len] = '\0';
  if(debug) fprintf(stderr, "Server: %s\n", str);
  //fprintf(stderr, "%s\n", str);
  

#ifdef WIN32
  closesocket(sock);
  WSACleanup( );
#else
  close(sock);
#endif

  Json::Reader reader;
  Json::Value ret;
  if(!reader.parse(str, ret))
	  ret = Json::Value::null;
  free(str);
  return ret;
}

void JsonRpcServer::ProcessRequest(SOCKET  newsockfd, char *buf) {
  Json::Value root;
  Json::Value response;
  Json::Reader reader;

  response["jsonrpc"] = "2.0";
  if(root.isMember("id")) response["id"] = root["id"];
  //fprintf(stderr, "Client: %s\n", buf);
  
  if(!reader.parse(buf, root)) {
    response["error"] = "Error parsing request";  ReturnJsonResponse(newsockfd, response);
  } else {
    if(!root.isMember("jsonrpc") || root.get("jsonrpc", "").asString() != "2.0") {
      response["error"] = "Error parsing request, parameter \"jsonrpc\" must be \"2.0\""; 
      ReturnJsonResponse(newsockfd, response);
    } else if(!root.isMember("method")) {
      response["error"] = "Error parsing request, parameter \"method\" not found"; 
      ReturnJsonResponse(newsockfd, response);
    } else {
      char name[1000];
      strcpy(name, root.get("method", "").asString().c_str());
      JsonRpcMethodAbstract *func = FindMethod(name);
      if(!func) {
	char str[1000]; sprintf(str, "Error parsing request, \"method\" %s not found", name);
	response["error"] = str; 
	ReturnJsonResponse(newsockfd, response);
      } else {
	// TODO: Invoke this as a thread
	func->Invoke(root, response);
	ReturnJsonResponse(newsockfd, response);
      }
    }
  }
}

#define BOUNDARY_STRING "---------------------------3767824881932084641640062629"
#define SEND_RQ(MSG) \
  send(sock,MSG,strlen(MSG),0);

char *http_request(const char *hostname, const char *remote_file, int *err, const char *request_type,
                   const char *type, const char *post_parameters, int postSz, const char *host_referer,
                   const char *referer, const char *cookie, bool getHeader, int *sz) {
  if(sz) *sz = 0;
  *err = 0;

#ifdef WIN32
  {
    WSADATA WsaData;
    WSAStartup (0x0101, &WsaData);
  }
#endif

  sockaddr_in sin;
  int sock = socket (AF_INET, SOCK_STREAM, 0);
  if (sock == -1) {
    if(err) *err = -100;
    return NULL;
  }
  sin.sin_family = AF_INET;
  sin.sin_port = htons( (unsigned short)80);

  struct hostent *host_addr = gethostbyname(hostname);
  if(host_addr==NULL) {
#ifdef WIN32
    closesocket(sock);
    WSACleanup( );
#else
    close(sock);
#endif
    if(err) *err = -103;
    return NULL;
  }
  sin.sin_addr.s_addr = *((int*)*host_addr->h_addr_list) ;

  if( connect (sock,(const struct sockaddr *)&sin, sizeof(sockaddr_in) ) == -1 ) {
#ifdef WIN32
  closesocket(sock);
  WSACleanup( );
#else
  close(sock);
#endif
    if(err) *err = -101;
    return NULL;
  }

  char req[1000]; sprintf(req, "%s ", request_type);
  SEND_RQ(req);
  SEND_RQ(remote_file);
  SEND_RQ(" HTTP/1.0\r\n");
  SEND_RQ("Accept: */*\r\n");
  SEND_RQ("User-Agent: Mozilla/4.0\r\n");
  SEND_RQ("Accept-Language: en-us\r\n");
  if(host_referer) { SEND_RQ("Host: "); SEND_RQ(host_referer ? host_referer : "vasuki.ucsd.edu"); SEND_RQ("\r\n") };
  if(referer) { SEND_RQ("Referer: "); SEND_RQ(referer); SEND_RQ("\r\n"); }

  if(post_parameters) {
    if(type && (!strcmp(type,"form") || !strcmp(type,"binary"))) { SEND_RQ("Content-Type: multipart/form-data; boundary="); SEND_RQ(BOUNDARY_STRING); SEND_RQ("\r\n"); }
    else if(type && !strcmp(type,"url")) { SEND_RQ("Content-Type: application/x-www-form-urlencoded\r\n"); }
    else { SEND_RQ("Content-Type: text/xml; charset=ISO8859-15\r\n"); }
  }
  if(cookie) { SEND_RQ("Cookie: "); SEND_RQ(cookie); SEND_RQ("\r\n"); }
  char content_header[100];
  if(post_parameters) {
    sprintf(content_header,"Content-Length: %d\r\n",(postSz ? postSz : (int)strlen(post_parameters))+2);
    SEND_RQ(content_header);
  }

  SEND_RQ("\r\n");
  if(post_parameters) {
    if(postSz)
      send(sock,post_parameters,postSz,0);
    else {
      SEND_RQ(post_parameters);
    }
  }
  SEND_RQ("\r\n----\r\n");


  char c1[2] = {0,0};
  int l,line_length = 0;
  bool loop = true;
  bool bHeader = false;
  char header[1000];

  int len = 0;
  char *message=NULL;
  strcpy(header, "");
  int res = 1;
  float v;
  char *ptr;
  while(loop && line_length < 999) {
    if(getHeader && c1[0]) {
      message = (char*)realloc(message, len + 2);
      message[len++] = c1[0];
    }

    l = recv(sock, c1, 1, 0);
    //fprintf(stderr, "%c", c1[0]);
    if(l<0) loop = false;
    if(c1[0]=='\n') {
      if(line_length == 0) loop = false;

      line_length = 0;
      if((ptr=strstr(header, "HTTP")))
        if(sscanf(ptr, "HTTP/%f %d", &v, &res) == 2) {
          if(res >= 200 && res < 300) {
            bHeader = true;
            *err = res;
          } else
            *err = -res;
        }

      strcpy(header, "");
    }
    else if(c1[0]!='\r') line_length++;
    strcat(header, c1);
  }

  if(bHeader) {
    char p[1024];
    while((l = recv(sock,p,1023,0)) > 0)  {
      p[l] = '\0';
      message = (char*)realloc(message, len + l + 200002);
      memcpy(message+len, p, l);
      message[len+l] = '\0';
      len += l;
    }
  } else {
#ifdef WIN32
  closesocket(sock);
  WSACleanup( );
#else
  close(sock);
#endif
    if(err) *err = -102;
    return message;
  }
  if(sz) *sz = len;


#ifdef WIN32
  closesocket(sock);
  WSACleanup( );
#else
  close(sock);
#endif

  return message;
}
