function response = jsonrpc_request(host, port, query)
    query.jsonrpc = '2.0';
    queryStr = savejson(query)
    str = tcp_request(host, port, queryStr);
    response = loadjson(str);
end