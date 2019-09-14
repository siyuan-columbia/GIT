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
#include <ctype.h>

#define BUFSIZE 1219
#define ERROR -1
#define USAGE                                                \
    "usage:\n"                                               \
    "  transferclient [options]\n"                           \
    "options:\n"                                             \
    "  -s                  Server (Default: localhost)\n"    \
    "  -p                  Port (Default: 19121)\n"           \
    "  -o                  Output file (Default cs6200.txt)\n" \
    "  -h                  Show this help message\n"

/* OPTIONS DESCRIPTOR ====================================================== */
static struct option gLongOptions[] = {
    {"server", required_argument, NULL, 's'},
    {"port", required_argument, NULL, 'p'},
    {"output", required_argument, NULL, 'o'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}};

/* Main ========================================================= */
int main(int argc, char **argv)
{
    int option_char = 0;
    char *hostname = "localhost";
    unsigned short portno = 19121;
    char *filename = "cs6200.txt";
    long nHostAddress;
    struct hostent* pHostInfo;
    char buffer[BUFSIZE];

    setbuf(stdout, NULL);

    // Parse and set command line arguments
    while ((option_char = getopt_long(argc, argv, "s:p:o:hx", gLongOptions, NULL)) != -1)
    {
        switch (option_char)
        {
        case 's': // server
            hostname = optarg;
            break;
        case 'p': // listen-port
            portno = atoi(optarg);
            break;
        default:
            fprintf(stderr, "%s", USAGE);
            exit(1);
        case 'o': // filename
            filename = optarg;
            break;
        case 'h': // help
            fprintf(stdout, "%s", USAGE);
            exit(0);
            break;
        }
    }

    if (NULL == hostname)
    {
        fprintf(stderr, "%s @ %d: invalid host name\n", __FILE__, __LINE__);
        exit(1);
    }

    if (NULL == filename)
    {
        fprintf(stderr, "%s @ %d: invalid filename\n", __FILE__, __LINE__);
        exit(1);
    }

    if ((portno < 1025) || (portno > 65535))
    {
        fprintf(stderr, "%s @ %d: invalid port number (%d)\n", __FILE__, __LINE__, portno);
        exit(1);
    }

    /* Socket Code Here */
    struct sockaddr_in remote_server;
    int sock;

    if((sock=socket(AF_INET,SOCK_STREAM,0))==ERROR)
    {
        perror("socket");
        exit(-1);
    }
    remote_server.sin_family=AF_INET;
    
    
    remote_server.sin_port=htons(portno);
    
    
    pHostInfo=gethostbyname(hostname);
    memcpy(&nHostAddress,pHostInfo->h_addr,pHostInfo->h_length);
    
    remote_server.sin_addr.s_addr=nHostAddress;
    
    bzero(&remote_server.sin_zero,8);
    if((connect(sock,(struct sockaddr *)&remote_server,sizeof(struct sockaddr)))==ERROR)
    {
        perror("connect");
        exit(1);
    }
    //printf("geting file");
    
    FILE *fp;
    int ch=0;
    bzero(buffer,BUFSIZE);
    fp=fopen(filename,"a");
    int words=0;
    //printf("%d",words);
    read(sock,&words,sizeof(int));
    //printf("%d",words);
    while(ch!=words)
    {
        read(sock,buffer,BUFSIZE);
        //printf("%s\n",buffer);
        fprintf(fp,"%s ",buffer);
        ch++;
    }
    fclose(fp);
    ch='\0';
    close(sock);
    
}
