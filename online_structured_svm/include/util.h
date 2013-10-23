#ifndef __UTIL_H
#define __UTIL_H

#define HAVE_SSE 1

#if HAVE_SSE
#include <xmmintrin.h>
#define ALIGN(x) (float*)(((((long)x)+15)/16)*16)
#define IS_ALIGNED(x) (((long)x)%16==0)
#ifdef WIN32
  #define ALGNW __declspec(align(16))
  #define ALGNL
#else
  #define ALGNW
  #define ALGNL __attribute__((aligned(16)))
#endif
#endif

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <vector>
#include <map>

#ifdef WIN32
inline int posix_memalign(void **retval, int align, unsigned int sz) {
#if HAVE_SSE
	*retval = _aligned_malloc(sz, align);  // might cause problems on windows, because I haven't used _aligned_free().  Don't define HAVE_SSE
#else
	*retval = malloc(sz);
#endif
	return *retval == NULL;
}
#endif

#include "json/json.h"

/**
 * @file util.h
 * @brief Miscellaneous helper routines 
 */

// Define this to run in memory leak check mode on Windows.  Could also include <vld.h> in test application instead.
// In linux, can leak check a program by just running as valgrind ./program.out
//#define DEBUG_MEMORY_LEAKS  

#ifndef WIN32
#undef DEBUG_MEMORY_LEAKS
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif

#ifdef DEBUG_MEMORY_LEAKS
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif



#ifndef M_E
#define M_E 2.7182818284590452354   /**< The constant e: ln(x)=log_e(x) */
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846 /**< The constant PI */
#endif

#ifndef INFINITY
#define INFINITY HUGE_VAL  /**< The constant used to represent infinity */
#endif


#define my_max(x,y) ((x) > (y) ? (x) : (y)) /**< Take the max over two numbers x and y */
#define my_min(x,y) ((x) < (y) ? (x) : (y)) /**< Take the min over two numbers x and y */
#define my_abs(x) ((x) < 0 ? -(x) : (x)) /**< Take the absolute value of x */
#define my_round(x) (x < 0 ? ((int)(x-.5f)) : ((int)(x+.5f))) /**< Round a real number to the nearest integer */

#define SAFE(x, mi, ma) (my_min(ma,my_max(mi,x)))  /**< Bound x to be within the interval [mi,ma] */
#define SQR(x) ((x)*(x)) /**< Take the square of x */
#define LOG2(x) (log((double)x)/log(2.0))  /**< log_2(x) */
#define LOG_E(x) (log((double)x)/log(M_E))  /**< ln(x) */
#define LN(x) LOG_E(x) /**< ln(x) */
#define LOG_B(x,b) (log((double)x)/log((double)b))  /**< log_b(x) */

#define RAND_FLOAT (((float)rand())/RAND_MAX)  /**< Generate a random floating point between 0 and 1 */
#define RAND_DOUBLE (((double)rand())/RAND_MAX) /**< Generate a random double between 0 and 1 */
#define RAND_GAUSSIAN (sqrt(-2*LOG_E(RAND_DOUBLE))*cos(2*M_PI*RAND_DOUBLE)) /**< Generate a random number from a Gaussian distribution with mean 0 and standard deviation 1 */

#define SIGMOID2(t) (1.0f/(1+exp(-t)))  /**< Sigmoid function of t */

#ifndef isnan
#define isnan(x) ((x) != (x))  /**< check if something is not a number */
#endif

#ifdef WIN32
#define SLEEP(us) Sleep(us/1000)  /**< Sleep for us microseconds */
#else
#define SLEEP(us) usleep(us) /**< Sleep for us microseconds */
#endif

#ifndef WIN32
/// @cond
  #define stricmp strcasecmp
  #define strnicmp strncasecmp
/// @endcond
#endif


#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#include <tchar.h>
#include "json/stdint.h"
#else
#include <stdint.h>
#endif


/**
 * @brief Remove trailing spaces or carriage returns from a string
 * @param str The string (modified by this function)
 */
inline void chomp(char *str) { 
	int i = (int)strlen(str)-1; 
	while(i >= 0 && isspace(str[i])) {
		str[i--] = '\0';
	}
}

/**
 * @brief Remove preceding spaces or carriage returns from a string
 * @param str The string (modified by this function)
 */
inline void pre_chomp(char *str) { 
  int i = 0, n = 0;
  while(isspace(str[i])) i++;
  if(i) { 
    while(str[i]) {
     str[n++] = str[i];
     i++;
    }
    str[n] = '\0';
  }
}


/**
 * @brief Iteratively tokenize a string
 * @param str The input string (gets modified by this function)
 * @param delim An array of character delimeters used to tokenize the string
 * @param next A pointer to a string that is set by this function and should be passed to the subsequent call to my_strtok()
 * @return The extracted token
 */
inline char *my_strtok(char *str, const char *delim, char **next) {
  char *str2 = str ? str : *next;
  if(!str2) return NULL;
  int len = (int)strlen(str2);
  for(int i = 0; i < len; i++) {
    for(int j = 0; j < (int)strlen(delim); j++) {
      if(str2[i] == delim[j]) {
        str2[i] = '\0';
        *next = i < len-1 ? str2 + i + 1 : NULL;
        return str2;
      }
    }
  }
  *next = NULL;
  return str2;
}


/**
 * @brief Return a dynamically allocated copy of a string
 * @param str The string 
 * @return The copy of the string (it should be freed using StringFree()
 */
inline char *StringCopy(const char *str) { char *retval = (char*)malloc((strlen(str)+1)*sizeof(char)); strcpy(retval, str); return retval; }

/**
 * @brief Free a string created using StringCopy()
 * @param str The string 
 */
inline void StringFree(char *str) { free(str); }

/**
 * @brief Convert a string to upper case
 * @param str The string (modified by this function)
 */
inline void StringToUpperCase(char *str) {
  for(int i = 0; i < (int)strlen(str); i++)
    str[i] = toupper(str[i]);
}

/**
 * @brief Convert a string to lower case
 * @param str The string (modified by this function)
 */
inline void StringToLowerCase(char *str) {
  for(int i = 0; i < (int)strlen(str); i++)
    str[i] = tolower(str[i]);
}

/**
 * @brief Given a file path f1 from the current directory, get the relative path to f1 if we are in the folder f2 instead of in the current directory
 * @param f1 the source file path
 * @param f2 the alternate current path
 * @param out string into which the computed relative path is stored
 * @return out
 */
inline char *GetRelativePath(const char *f1, const char *f2, char *out) {
  int i;
  strcpy(out, "");

  if(f1[0] == '/' || f1[0] == '\\') {
    strcpy(out, f1); 
    return out; 
  }

  char rels[1000]; strcpy(rels, "");
  i = 0;
  bool isEq = true;
  int lastDirEq=0;
  while(f2[i]) {
    if(isEq && f1[i] != f2[i]) {
      isEq = false;
    }
    if(f2[i] == '/' && f2[i-1] != '/') {
      if(!isEq) strcat(rels, "../");
      else lastDirEq = i;
    }
    i++;
  }

  while(f1[lastDirEq] == '/')
    lastDirEq++;
  strcat(out, rels);
  strcat(out, f1+lastDirEq);
  return out;
}

/**
 * @brief Convert a relative path to a full path
 * @param relativePath The relative path
 * @param currDir The full path of the current directory
 * @param fullPath The full path that is set by this function
 * @return fullPath
 */
inline char *GetFullPath(const char *relativePath, const char *currDir, char *fullPath) {
  if(relativePath[0] == '/' || (relativePath[1] == ':' && 
				(relativePath[2] == '/' || relativePath[2] == '\\'))) {
    strcpy(fullPath, relativePath);   // path is already a full path
  } else {
    sprintf(fullPath, "%s/%s", currDir, relativePath);
  }
  return fullPath;
}


/**
 * @brief Extract the filename from a path (ignoring directories) 
 * @param str The path
 * @param fname The extracted filename
 */
inline void ExtractFilename(const char *str, char *fname) {
  int i = (int)strlen(str)-1;
  while(i >= 0 && str[i] != '/' && str[i] != '\\')
    i--;
  strcpy(fname, str+i+1);
}


/**
 * @brief Extract the folder and filename from a path 
 * @param str The path
 * @param folder The extracted folder
 * @param fname The extracted filename
 */
inline void ExtractFolderAndFileName(const char *str, char *folder, char *fname) {
  int i = (int)strlen(str)-1;
  while(i >= 0 && str[i] != '/' && str[i] != '\\')
    i--;
  strcpy(fname, str+i+1);
  
  if(i > 0) i--;
  int num = 0;
  while(i >= 0 && str[i] != '/' && str[i] != '\\') {
    i--;
    num++;
  }
  strcpy(folder, str+i+1);
  folder[num] = '\0';
}


/**
 * @brief Extract the filename from a path (ignoring directories) 
 * @param str The path
 * @param path The extracted path
 */
inline void ExtractPathname(const char *str, char *path) {
  int i = (int)strlen(str);
  while(i > 0 && str[i] != '/' && str[i] != '\\') {
    i--;
  }
  strcpy(path, str);
  path[i] = '\0';
}

/**
 * @brief Remove the file extension from a file name
 * @param str file name (modified by this function)
 */
inline void StripFileExtension(char *str) {
  int i = (int)strlen(str);
  while(i >= 0) {
    if(str[i] == '.') {
      strcpy(str+i, "");
      break;
    }
    i--;
  }
}

/**
 * @brief Get the file extension from a file name (the stuff after the '.')
 * @param str file name 
 * @return the file extension
 */
inline const char *GetFileExtension(const char *str) {
  int i = (int)strlen(str);
  while(i >= 0 && str[i] != '/') {
    if(str[i] == '.') {
      return str+i+1;
      break;
    }
    i--;
  }
  return str+strlen(str);
}

/**
 * @brief Replace all instances of a character in a string with another character
 * @param str The string (modified by this function)
 * @param c The character to be replaced
 * @param c2 The character that will replace all isntances of c
 */
inline void StringReplaceChar(char *str, char c, char c2) {
  for(int i = 0; i < (int)strlen(str); i++)
    if(str[i] == c)
      str[i] = c2;
}

inline char *StringReplace(const char *str, const char *match, const char *replace, bool replaceAll = true) {
  char *retval = StringCopy(str), *ptr, *retvalNew = NULL;
  char *tmp = StringCopy(str);
  int len = (int)strlen(retval)+1;
  int lenMatch = (int)strlen(match);
  int lenReplace = (int)strlen(replace);
  int curr = 0;
  while((ptr=strstr(retval+curr, match)) != NULL) {
    len += -lenMatch+lenReplace;
    retvalNew = (char*)realloc(retval, sizeof(char)*(len+lenMatch));
    curr = ((int)(ptr-retval)) + lenReplace;
    ptr = retvalNew + (ptr-retval);
    retval = retvalNew;
    strcpy(tmp, ptr + lenMatch);
    strcpy(ptr, replace);
    strcpy(ptr + lenReplace, tmp);
    if(!replaceAll)
      break;
  }
  free(tmp);
  return retval;
}

/**
 * @brief Tokenize a string
 * @param str The input string (gets modified by this function)
 * @param d An array of character delimeters used to tokenize the string
 * @param toks An array of strings into which the extracted tokens are stored (it is assumed that the array is allocated and of sufficient size before this function is called)
 * @return The number of extracted tokens in toks
 */
inline int SplitString(char *str, char **toks, const char *d) {
  int num = 0;
  char *ptr;
  while((ptr=strtok(str, d)) != NULL) {
    toks[num++] = ptr;
    str = NULL;
  }
  return num;
}

/**
 * @brief Check if a file name is safe, such that it should be safe to write to even if the filename was specified by a client over the web
 * @param fname the filename
 * @return true if the file name is safe
 */
inline bool IsSafeFileName(const char *fname) { 
  return !strstr(fname, "/") && !strstr(fname, "\\") && !strstr(fname, "~"); 
}

/**
 * @brief Check if a file exists
 * @param fname the filename
 * @return true if the file exists
 */
inline bool FileExists(const char *fname) {
  FILE *fin = fopen(fname, "r");
  if(fin) fclose(fin);
  return fin != NULL;
}




/**
 * @brief Iteratively read the next line from a multi-line string
 * @param buf the input string (modified by this function
 * @param n the maximum number of characters in the line
 * @param source a pointer to a string that is set by this function.  This should be passed to the subsequent call to sgets()
 * @return the extracted line
 */
inline char *sgets(char *buf, int n, char **source) {
  int k = 0;
  //fprintf(stderr, "sgets %s\n", *source);

  while (n-- > 1 && **source && **source != '\n')  {
    buf[k++] = *(*source)++;
  }
  buf[k] = 0;
  if (n < 1)
    return buf;
  if (!**source) {
    if (k <= 0)
      return NULL;
    buf[k++] = '\n';
    buf[k] = 0;
    //(*source)++;
    return buf;
  }
  buf[k++] = *(*source)++;
  buf[k] = 0;
  return buf;
}



/**
 * @brief Read the entire contents of a file into a dynamically allocated string
 * @param fname the file name
 * @return the contents of the file
 */
inline char *ReadStringFile(const char *fname) {
  unsigned long len = 0;
  char *retval = NULL;
  FILE *fin = fopen(fname, "r");
  if(!fin) return NULL;
  char line[30000];
  unsigned long alloc = 0;
  while(fgets(line, 29999, fin)) {
    if(len + ((unsigned long)strlen(line))+1 >= alloc) {
      alloc = (unsigned long)(alloc*1.1 + strlen(line) + 1024);
      retval = (char*)realloc(retval, alloc);
    }
    strcpy(retval + len, line);
    len += (unsigned long)strlen(line);
  }
  fclose(fin);
  return retval;
}

/**
 * @brief Read the entire contents of a file into a dynamically allocated buffer
 * @param fname the file name
 * @param len the number of bytes (the size of the returned array, set by this function)
 * @return the contents of the file
 */
inline unsigned char *ReadBinaryFile(const char *fname, int *len) {
  *len = 0;
  FILE *fin = fopen(fname, "rb");
  if(!fin) return NULL;
  int num;
  unsigned char *retval = (unsigned char*)malloc(1024);
  while((num=(int)fread(retval+*len, 1, 1024, fin)) > 0) {
    retval = (unsigned char*)realloc(retval, *len + num + 1024);
    *len += num;
  }
  fclose(fin);
  return retval;
}



/**
 * @brief Compute a random permutation of the numbers between 0 and num-1
 * @param num The number of elements in the random permutation
 * @return An allocated array of the random permutation
 */
inline int *RandPerm(int num) {
  int i, ind, tmp;
  int *a = (int*)malloc(num*sizeof(int));
  for(i = 0; i < num; i++)
    a[i] = i;
  for(i = 0; i < num; i++) {
    ind = i + rand()%(num-i); 
    tmp = a[i];
    a[i] = a[ind];
    a[ind] = tmp;
  }
  return a;
}

/**
 * @brief Compute a random split of numTotal elements, such that num0 elements are  assigned to 0 and numTotal-num0 elements are assigned 1
 * @param num0 The number of elements in the random split that are assigned 0
 * @param numTotal The total number of elements in the random split 
 * @return An allocated array of size numTotal, where each element is either 0 or 1
 */
inline int *RandSplit(int num0, int numTotal) {
  int *perm = RandPerm(numTotal);
  int *retval = (int*)malloc(numTotal*sizeof(int));
  int i;
  for(i = 0; i < num0; i++)
    retval[perm[i]] = 0;
  for(; i < numTotal; i++)
    retval[perm[i]] = 1;
  free(perm);
  return retval;
}


#define TALIGN1(d) d = d + (16/sizeof(T))-(d%(16/sizeof(T)))
#define TALIGN2(d) d = d + (16/sizeof(T*))-(d%(16/sizeof(T*)))

inline void FreeArray(void *a, bool align = false) {
#ifdef WIN32
#if HAVE_SSE
  if(align) { _aligned_free(a); return; }
#endif
#endif
  free(a);
}

/**
 * @brief Dynamically allocate a 1d array (it should be freed using free())
 * @param r The number of elements in the array
 * @return An allocated array
 */
template <typename T>
inline T *Create1DArray(int r, bool align=false) {
  T *retval;
  if(align) {
    TALIGN1(r);
    int e = posix_memalign((void**)&retval, 16, sizeof(T)*r);  assert(!e);
  } else 
    retval = (T*)malloc(sizeof(T)*r);
  for(int i = 0; i < r; i++) 
    retval[i] = 0;
  return retval;
}

/**
 * @brief Dynamically allocate a 2d rXc array (it should be freed using free())
 * @param r The number of rows in the array
 * @param c The number of columns in the array
 * @return An allocated array
 */
template <typename T>
inline T **Create2DArray(int r, int c, bool align=false) {
  T **retval;
  if(align) {
    TALIGN2(r);  TALIGN1(c);
    int e = posix_memalign((void**)&retval, 16, (sizeof(T*)+sizeof(T)*c)*r);  assert(!e);
  } else 
    retval = (T**)malloc((sizeof(T*)+sizeof(T)*c)*r);
  T *ptr = (T*)(retval+r);
  for(int i = 0; i < r; i++, ptr += c) {
    retval[i] = ptr;
    for(int j = 0; j < c; j++) 
      retval[i][j] = 0;
  }
  return retval;
}

/**
 * @brief Dynamically allocate a 3d d1Xd2xd3 array (it should be freed using free())
 * @param d1 the 1st dimension size
 * @param d2 the 2nd dimension size
 * @param d3 the 3rd dimension size
 * @return An allocated array
 */
template <typename T>
inline T ***Create3DArray(int d1, int d2, int d3, bool align=false) {
  T ***retval;
  if(align) {
    TALIGN2(d1);  TALIGN2(d2);  TALIGN1(d3);
    int e = posix_memalign((void**)&retval, 16, (sizeof(T**)+(sizeof(T*)+sizeof(T)*d3)*d2)*d1);  assert(!e);
  } else 
    retval = (T***)malloc((sizeof(T**)+(sizeof(T*)+sizeof(T)*d3)*d2)*d1);

  T **ptr = (T**)(retval+d1);
  T *ptr2 = (T*)(ptr+d1*d2);
  for(int i = 0; i < d1; i++, ptr += d2) {
    retval[i] = ptr;
    for(int j = 0; j < d2; j++, ptr2 += d3) {
      retval[i][j] = ptr2;
      if(j == 0) {
	for(int k = 0; k < d3; k++) 
	  retval[i][j][k] = 0;
      } else
	memcpy(retval[i][j], retval[i][0], sizeof(T)*d3);
    }
  }
  return retval;
}

/**
 * @brief Dynamically allocate a 4d d1Xd2xd3xd4 array (it should be freed using free())
 * @param d1 the 1st dimension size
 * @param d2 the 2nd dimension size
 * @param d3 the 3rd dimension size
 * @param d4 the 4th dimension size
 * @return An allocated array
 */
template <typename T>
inline T ****Create4DArray(int d1, int d2, int d3, int d4, bool align = false) {
  T ****retval;
  if(align) {
    TALIGN2(d1);  TALIGN2(d2);  TALIGN2(d3);  TALIGN1(d4);
    int e = posix_memalign((void**)&retval, 16, (sizeof(T***)+(sizeof(T**)+(sizeof(T*)+sizeof(T)*d4)*d3)*d2)*d1);  assert(!e);
  } else 
    retval = (T****)malloc((sizeof(T***)+(sizeof(T**)+(sizeof(T*)+sizeof(T)*d4)*d3)*d2)*d1);

  T ***ptr = (T***)(retval+d1);
  T **ptr2 = (T**)(ptr+d1*d2);
  T *ptr3 = (T*)(ptr+d1*d2*d3);
  for(int i = 0; i < d1; i++, ptr += d2) {
    retval[i] = ptr;
    for(int j = 0; j < d2; j++, ptr2 += d3) {
      retval[i][j] = ptr2;
      for(int k = 0; k < d3; k++, ptr3 += d4) {
	retval[i][j][k] = ptr3;
	if(k == 0) {
	  for(int l = 0; l < d4; l++) 
	    retval[i][j][k][l] = 0;
	} else
	  memcpy(retval[i][j][k], retval[i][j][0], sizeof(T)*d4);
      }
    }
  }
  return retval;
}

/**
 * @brief Write a 1d array to file
 * @param m an array of size d1
 * @param fout the file handle
 * @param d1 The number of elements in the array
 * @param name a string identifier of the array, to be stored along with the data
 * @param header an additional string to be stored along with the data
 * @return true on success
 */
template <typename T>
inline bool Save1DArray(FILE *fout, T *m, int d1, const char *name=NULL, const char *header=NULL) {
  int n = 1, c = sizeof(T), l = name ? strlen(name) : 0, l2 = header ? strlen(header) : 0;
  return fwrite(&l, sizeof(int), 1, fout) &&
    (!l || fwrite(name, sizeof(char), l, fout)) &&
    fwrite(&l2, sizeof(int), 1, fout) &&
    (!l2 || fwrite(header, sizeof(char), l2, fout)) &&
    fwrite(&n, sizeof(int), 1, fout) &&
    fwrite(&d1, sizeof(int), 1, fout) &&
    fwrite(&c, sizeof(int), 1, fout) &&
    fwrite(m, sizeof(T), d1, fout);
}



/**
 * @brief Write a 2d array to file
 * @param m an array of size d1Xd2
 * @param fout the file handle
 * @param d1 the 1st dimension size
 * @param d2 the 2nd dimension size
 * @param name a string identifier of the array, to be stored along with the data
 * @param header an additional string to be stored along with the data
 * @return true on success
 */
template <typename T>
inline bool Save2DArray(FILE *fout, T **m, int d1, int d2, const char *name, const char *header) {
  int n = 2, c = sizeof(T), l = name ? strlen(name) : 0, l2 = header ? strlen(header) : 0;
  if(!(fwrite(&l, sizeof(int), 1, fout) &&
       (!l || fwrite(name, sizeof(char), l, fout)) &&
       fwrite(&l2, sizeof(int), 1, fout) &&
       (!l2 || fwrite(header, sizeof(char), l2, fout)) &&
       fwrite(&n, sizeof(int), 1, fout) &&
       fwrite(&d1, sizeof(int), 1, fout) &&
       fwrite(&d2, sizeof(int), 1, fout) &&
       fwrite(&c, sizeof(int), 1, fout)))
    return false;
  for(int i = 0; i < d1; i++)
    if(!fwrite(m[i], sizeof(T), d2, fout))
      return false;
  return true;
}

/**
 * @brief Write a 3d array to file
 * @param m an array of size d1Xd2Xd3
 * @param fout the file handle
 * @param d1 the 1st dimension size
 * @param d2 the 2nd dimension size
 * @param d3 the 3rd dimension size
 * @param name a string identifier of the array, to be stored along with the data
 * @param header an additional string to be stored along with the data
 * @return true on success
 */
template <typename T>
inline bool Save3DArray(FILE *fout, T ***m, int d1, int d2, int d3, const char *name, const char *header) {
  int n = 3, c = sizeof(T), l = name ? strlen(name) : 0, l2 = header ? strlen(header) : 0;
  if(!(fwrite(&l, sizeof(int), 1, fout) &&
       (!l || fwrite(name, sizeof(char), l, fout)) &&
       fwrite(&l2, sizeof(int), 1, fout) &&
       (!l2 || fwrite(header, sizeof(char), l2, fout)) &&
       fwrite(&n, sizeof(int), 1, fout) &&
       fwrite(&d1, sizeof(int), 1, fout) &&
       fwrite(&d2, sizeof(int), 1, fout) &&
       fwrite(&d3, sizeof(int), 1, fout) &&
       fwrite(&c, sizeof(int), 1, fout)))
    return false;
  for(int i = 0; i < d1; i++)
    for(int j = 0; j < d2; j++)
      if(!fwrite(m[i][j], sizeof(T), d3, fout))
	return false;
  return true;
}


/**
 * @brief Write a 4d array to file
 * @param m an array of size d1Xd2Xd3Xd4
 * @param fout the file handle
 * @param d1 the 1st dimension size
 * @param d2 the 2nd dimension size
 * @param d3 the 3rd dimension size
 * @param d4 the 4th dimension size
 * @param name a string identifier of the array, to be stored along with the data
 * @param header an additional string to be stored along with the data
 * @return true on success
 */
template <typename T>
inline bool Save4DArray(FILE *fout, T ****m, int d1, int d2, int d3, int d4, const char *name=NULL, const char *header=NULL) {
  int n = 4, c = sizeof(T), l = name ? strlen(name) : 0, l2 = header ? strlen(header) : 0;
  if(!(fwrite(&l, sizeof(int), 1, fout) &&
       (!l || fwrite(name, sizeof(char), l, fout)) &&
       fwrite(&l2, sizeof(int), 1, fout) &&
       (!l2 || fwrite(header, sizeof(char), l2, fout)) &&
       fwrite(&n, sizeof(int), 1, fout) &&
       fwrite(&d1, sizeof(int), 1, fout) &&
       fwrite(&d2, sizeof(int), 1, fout) &&
       fwrite(&d3, sizeof(int), 1, fout) &&
       fwrite(&d4, sizeof(int), 1, fout) &&
       fwrite(&c, sizeof(int), 1, fout)))
    return false;
  for(int i = 0; i < d1; i++)
    for(int j = 0; j < d2; j++)
      for(int k = 0; k < d3; k++)
	if(!fwrite(m[i][j][k], sizeof(T), d4, fout))
	  return false;
  return true;
}

template <typename T>
inline void *LoadNDArray(FILE *fin, int *dims, int *numDims, char *name=NULL, char *header=NULL) {
  char nameb[1000], headerb[1000];
  int n = 1, c = sizeof(T), l = name ? strlen(name) : 0, l2 = header ? strlen(header) : 0, d1;
  if(!(fread(&l, sizeof(int), 1, fin) &&
       (!l || fread(nameb, sizeof(char), l, fin)) &&
       fread(&l2, sizeof(int), 1, fin) &&
       (!l2 || fread(headerb, sizeof(char), l2, fin)) &&
       fread(&n, sizeof(int), 1, fin) && n > 0 && n <= 4))
    return NULL;
  int sz = 1;
  for(int i = 0; i < n; i++) {
    fread(&dims[i], sizeof(int), 1, fin);
    sz *= dims[i];
  }
  if(name) strcpy(name, nameb);
  if(header) strcpy(header, headerb);
  if(numDims) *numDims = n;
  
  fread(&c, sizeof(int), 1, fin);

  if(n == 1) { 
    T *r = Create1DArray<T>(dims[0]); 
    if(!fread(r, sizeof(T), dims[0], fin))
      return NULL;
    return r;
  } else if(n == 2) { 
    T **r = Create2DArray<T>(dims[0],dims[1]); 
    for(int i = 0; i < dims[0]; i++)
      if(!fread(r[i], sizeof(T), dims[1], fin))
	return NULL;
    return r;
  } else if(n == 3) { 
    T ***r = Create3DArray<T>(dims[0],dims[1],dims[2]); 
    for(int i = 0; i < dims[0]; i++)
      for(int j = 0; j < dims[1]; j++)
	if(!fread(r[i][j], sizeof(T), dims[2], fin))
	  return NULL;
    return r;
  } else if(n == 4) { 
    T ****r = Create4DArray<T>(dims[0],dims[1],dims[2],dims[3]); 
    for(int i = 0; i < dims[0]; i++)
      for(int j = 0; j < dims[1]; j++)
	for(int k = 0; k < dims[2]; k++)
	  if(!fread(r[i][j][k], sizeof(T), dims[3], fin))
	    return NULL;
    return r;
  }
  return NULL;
}

/**
 * @brief Write a matlab script to file, which is capable of reading a file generated using a set of sequential calls to SaveXDArray()
 * @param fname the filename
 */
inline void SaveMatlabImport(const char *fname) {
  char fname2[1000];
  ExtractFilename(fname, fname2);
  StripFileExtension(fname2);
  strcat(fname2, ".bin");

  FILE *fout = fopen(fname, "w");
  assert(fout);
  fprintf(fout, "fid=fopen('%s', 'r');\n", fname2);
  fprintf(fout, "num_mats=fread(fid,1,'int32');\n");
  fprintf(fout, "for t=1:num_mats\n");
  fprintf(fout, "  mat_l=fread(fid,1,'int32');\n");
  fprintf(fout, "  mat_name=fread(fid,mat_l,'schar');\n");
  fprintf(fout, "  mat_l2=fread(fid,1,'int32');\n");
  fprintf(fout, "  mat_header=fread(fid,mat_l2,'schar');\n");
  fprintf(fout, "  mat_n=fread(fid,1,'int32');\n");
  fprintf(fout, "  mat_d=fread(fid,mat_n,'int32');\n");
  fprintf(fout, "  mat_c=fread(fid,1,'int32');\n");
  fprintf(fout, "  mat=zeros(mat_d');\n");
  fprintf(fout, "  if mat_n==1, mat=fread(fid,mat_d(1),char(mat_header'));\n");
  fprintf(fout, "  elseif mat_n==2, for j=1:mat_d(1) mat(j,:)=fread(fid,mat_d(2),char(mat_header')); end\n");
  fprintf(fout, "  elseif mat_n==3, for j=1:mat_d(1) for k=1:mat_d(2) mat(j,k,:)=fread(fid,mat_d(3),char(mat_header')); end; end\n");
  fprintf(fout, "  elseif mat_n==4, for j=1:mat_d(1) for k=1:mat_d(2) for l=1:mat_d(3) mat(j,k,l,:)=fread(fid,mat_d(4),char(mat_header')); end; end; end\n");
  fprintf(fout, "  end\n");
  fprintf(fout, "  assignin('base', char(mat_name'), mat);\n");
  fprintf(fout, "end\n");
  fprintf(fout, "clearvars fid num_mats mat_l mat_name mat_l2 mat_header mat_n mat_d mat_c mat t j k\n");
  fclose(fout);
}

/**
 * @brief Create a directory if it doesn't exist already
 * @param dirName The directory name
 * @param permissions if non-zero the permissions given to the created directory
 */
#ifndef WIN32
inline void CreateDirectoryIfNecessary(const char *dirName, mode_t permissions=0775) {
  struct stat st;
  umask(0);
  if(stat(dirName, &st) != 0) 
    if(mkdir(dirName, permissions) != 0)
      fprintf(stderr, "mkdir %s failed\n", dirName);
}
#else
inline void CreateDirectoryIfNecessary(const char *dirName, int permissions=0) {
  CreateDirectory(dirName, NULL);
}
#endif


inline bool DirectoryExists(const char *dir)
{
#ifdef WIN32 
  DWORD ftyp = GetFileAttributesA(dir);
  return ftyp != INVALID_FILE_ATTRIBUTES && (ftyp & FILE_ATTRIBUTE_DIRECTORY);
#else
  struct stat sb;
  return (stat(dir, &sb) == 0 && S_ISDIR(sb.st_mode));
#endif
}



/**
 * @brief List all files in a directory whose file name contain a specified string
 * @param dirName The directory name
 * @param match The string we are searching for.  A '|' character can be used to specify multiple search strings (e.g., ".jpg|.png" will find all jpg or png files in a directory)
 * @param outName The list of files is written to a text file specified by outName
 */ 
inline char **ScanDir(const char *dirName, const char *match, int *numF) {
  char *ptr;
  int numMatches=0;
  char *matches[1000], tmpMatch[1000];
  char **files = (char**)malloc(sizeof(char*)), *buff = NULL;
  files[0] = NULL;
  int numFiles = 0;
  unsigned long sz = 0;
  strcpy(tmpMatch, match);
  while((ptr=strtok(numMatches ? NULL : tmpMatch,"|"))) 
    matches[numMatches++] = ptr;

#ifndef WIN32
  struct dirent *de=NULL;
  DIR *d = opendir(dirName);
  if(d == NULL) 
    fprintf(stderr, "Couldn't open directory %s\n", dirName);
  else {
    while((de = readdir(d)) != NULL) {
      for(int i = 0; i < numMatches; i++) {
	bool isDir = !strcmp(matches[i], "dir");
	char fname[1000];
	sprintf(fname, "%s/%s", dirName, de->d_name);
	if((isDir && DirectoryExists(fname)) || (!isDir && strstr(de->d_name, matches[i])) || !strcmp(matches[i], "*")) {
	  files = (char**)realloc(files, sizeof(char*)*(numFiles+2));
	  buff = (char*)realloc(buff, sizeof(char)*(sz+strlen(de->d_name)+1));
	  char *curr = buff+sz;
	  strcpy(curr, de->d_name);
	  files[numFiles++] = (char*)sz;
	  sz += strlen(curr)+1;
	  break;
	}
      }
    }
  }
  closedir(d);
#else
   WIN32_FIND_DATA search_data;
   memset(&search_data, 0, sizeof(WIN32_FIND_DATA));

   char dir[1000];
   sprintf(dir, "%s/*", dirName);
   HANDLE handle = FindFirstFile(dir, &search_data);
   while(handle != INVALID_HANDLE_VALUE) {
     for(int i = 0; i < numMatches; i++) {
       bool isDir = !strcmp(matches[i], "dir");
       char fname[1000];
       sprintf(fname, "%s/%s", dirName, search_data.cFileName);
       if((isDir && DirectoryExists(fname)) || (!isDir && strstr(search_data.cFileName, matches[i])) || !strcmp(matches[i], "*")) {
	 files = (char**)realloc(files, sizeof(char*)*(numFiles+2));
	 buff = (char*)realloc(buff, sizeof(char)*(sz+strlen(search_data.cFileName)+1));
	 char *curr = buff+sz;
	 strcpy(curr, search_data.cFileName);
	 files[numFiles++] = (char*)sz;
	 sz += (int)strlen(curr)+1;
	 break;
       }
     }
	 if(FindNextFile(handle, &search_data) == FALSE)
	   break;
   }
   ::FindClose(handle);
#endif

   if(buff) {
     files = (char**)realloc(files, sizeof(char*)*(numFiles+2) + sizeof(char)*sz);
     char *ptr2 = (char*)(files+numFiles+1);
     memcpy(ptr2, buff, sz);
     for(int i = 0; i < numFiles; i++) 
       files[i] = ptr2+(long)files[i];
     files[numFiles] = 0;
     free(buff);
   }
   *numF = numFiles;

  return files;

  //char str[400];
  //char tmpMatch[1000];
  //char *ptr;
  //int numMatches=0;
  //strcpy(tmpMatch, match);
  //while((ptr=strtok(numMatches ? NULL : tmpMatch,"|"))) {
  //if(numMatches) sprintf(str, "ls %s/*%s >> %s", dirName, ptr, outName);
  //  else sprintf(str, "ls %s/*%s > %s", dirName, ptr, outName);
  //if(system(str) < 0) fprintf(stderr, "ScanDir failed\n");
  //}
}

#endif
