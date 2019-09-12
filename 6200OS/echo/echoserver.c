#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <netdb.h>
#include <getopt.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <arpa/inet.h>


#define ERROR -1

#define BUFSIZE 1219

#define USAGE                                                                 \
"usage:\n"                                                                    \
"  echoserver [options]\n"                                                    \
"options:\n"                                                                  \
"  -p                  Port (Default: 19121)\n"                                \
"  -m                  Maximum pending connections (default: 1)\n"            \
"  -h                  Show this help message\n"                              \

/* OPTIONS DESCRIPTOR ====================================================== */
static struct option gLongOptions[] = {
  {"port",          required_argument,      NULL,           'p'},
  {"maxnpending",   required_argument,      NULL,           'm'},
  {"help",          no_argument,            NULL,           'h'},
  {NULL,            0,                      NULL,             0}
};


int main(int argc, char **argv) {
  int option_char;
  int portno = 19121; /* port to listen on */
  int maxnpending = 1;
  
  // Parse and set command line arguments
  while ((option_char = getopt_long(argc, argv, "p:m:hx", gLongOptions, NULL)) != -1) {
   switch (option_char) {
      case 'p': // listen-port
        portno = atoi(optarg);
        break;                                        
      default:
        fprintf(stderr, "%s ", USAGE);
        exit(1);
      case 'm': // server
        maxnpending = atoi(optarg);
        break; 
      case 'h': // help
        fprintf(stdout, "%s ", USAGE);
        exit(0);
        break;
    }
  }

    setbuf(stdout, NULL); // disable buffering

    if ((portno < 1025) || (portno > 65535)) {
        fprintf(stderr, "%s @ %d: invalid port number (%d)\n", __FILE__, __LINE__, portno);
        exit(1);
    }
    if (maxnpending < 1) {
        fprintf(stderr, "%s @ %d: invalid pending count (%d)\n", __FILE__, __LINE__, maxnpending);
        exit(1);
    }


  /* Socket Code Here */
  struct sockaddr_in server;
  struct sockaddr_in client;
  int sock;
  int new;
  int sockaddr_len=sizeof(struct sockaddr_in);
  int data_len;
  char data[BUFSIZE];
  
  /*
  if ((sock=socket(AF_INET,SOCK_STREAM,0))==ERROR)
  {
	  perror("server socket: ");
	  exit(-1);
  }
  */
  /*server structure*/
  server.sin_family=AF_INET;
  server.sin_port=htons(atoi(argv[1]));
  server.sin_addr.s_addr=INADDR_ANY;
  bzero(&server.sin_zero,8);
  
  /*
  if((bind(sock,(struct sockaddr *)&server, sockaddr_len))==ERROR)
  {
	  perror("bind: ");
	  exit(-1);
  }
  */
   /*main loop*/
   while(1)
   {
	   /*
	   if((new = accept(sock,(struct sockaddr *)&client, &sockaddr_len))==ERROR)
	   {
		   perror("accept");
		   exit(-1);
	   }
	   printf("New Client connected from port no %d abd IP %s\n",ntohs(client.sin_port),inet._ntoa(client.sin_addr));
	   data_len=1;
	   */
	   /*start a loop to tell a client is connected*/
	   while(data_len)
	   {
		   data_len=recv(new, data,BUFSIZE,0);
		   
		   if(data_len)
		   {
			   send(new,data,data_len,0);
			   data[data_len]='\0';
			   /*printf("sent msg:%s",data);*/
			   printf("%s",data);
		   }
	   }
	   /*printf("Client disconnected\n");*/
	   close(new)
   }
  


}
