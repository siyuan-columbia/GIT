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


#define BUFSIZE 1219
#define ERROR -1
#define USAGE                                                \
    "usage:\n"                                               \
    "  transferserver [options]\n"                           \
    "options:\n"                                             \
    "  -f                  Filename (Default: 6200.txt)\n" \
    "  -h                  Show this help message\n"         \
    "  -p                  Port (Default: 19121)\n"

/* OPTIONS DESCRIPTOR ====================================================== */
static struct option gLongOptions[] = {
    {"filename", required_argument, NULL, 'f'},
    {"help", no_argument, NULL, 'h'},
    {"port", required_argument, NULL, 'p'},
    {NULL, 0, NULL, 0}};

int main(int argc, char **argv)
{
    int option_char;
    int portno = 19121;             /* port to listen on */
    char *filename = "6200.txt"; /* file to transfer */
    int maxnpending = 1;
    char buffer[BUFSIZE];


    setbuf(stdout, NULL); // disable buffering

    // Parse and set command line arguments
    while ((option_char = getopt_long(argc, argv, "p:hf:x", gLongOptions, NULL)) != -1)
    {
        switch (option_char)
        {
        case 'p': // listen-port
            portno = atoi(optarg);
            break;
        default:
            fprintf(stderr, "%s", USAGE);
            exit(1);
        case 'h': // help
            fprintf(stdout, "%s", USAGE);
            exit(0);
            break;
        case 'f': // file to transfer
            filename = optarg;
            break;
        }
    }


    if ((portno < 1025) || (portno > 65535))
    {
        fprintf(stderr, "%s @ %d: invalid port number (%d)\n", __FILE__, __LINE__, portno);
        exit(1);
    }
    
    if (NULL == filename)
    {
        fprintf(stderr, "%s @ %d: invalid filename\n", __FILE__, __LINE__);
        exit(1);
    }

    /* Socket Code Here */
    
    struct sockaddr_in server;
    struct sockaddr_in client;
    int sock;
    int new;
    socklen_t sockaddr_len=sizeof(struct sockaddr_in);
   
    
    
    if ((sock=socket(AF_INET,SOCK_STREAM,0))==ERROR)
    {
        perror("server socket: ");
        exit(-1);
    }
    
    /*server structure*/
    server.sin_family=AF_INET;
    server.sin_port=htons(portno);
    server.sin_addr.s_addr=INADDR_ANY;
    bzero(&server.sin_zero,8);
    
    int option = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));
    
    if((bind(sock,(struct sockaddr *)&server, sockaddr_len))==ERROR)
    {
        perror("bind: ");
        exit(-1);
    }
    if((listen(sock,maxnpending))==ERROR)
    {
        perror("listen");
        exit(-1);
    }
    
    FILE *f;
    
    char c;
    char ch;
    /*main loop*/
    while(1)
    {
        bzero(buffer,BUFSIZE);
        int words=0;
        if((new = accept(sock, (struct sockaddr *) &client, &sockaddr_len))==ERROR)
        {
            perror("accept");
            exit(-1);
        }
        
        f=fopen(filename,"r");
        while((c=getc(f))!=EOF)
        {
            fscanf(f,"%s",buffer);
            words++;
        }
        write(new,&words,sizeof(int));
        rewind(f);
        
        while(ch!=EOF)
        {
            fscanf(f,"%s",buffer);
            //printf("%s\n",buffer);
            write(new,buffer,BUFSIZE);
            ch=fgetc(f);
        }
        ch='\0';
        fclose(f);
        close(new);
     
    }

}
