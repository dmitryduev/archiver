### Setting up nginx/gunicorn on RHEL7

The `nginx` server acts as a reverse proxy by forwarding incoming requests 
to the `gunicorn` WSGI-server, which in turn runs 
the `flask` python application. The `gunicorn` processes are managed by
`supervisord`, which makes sure are always up and running.
  
#### Installing `nginx`
```
sudo vi /etc/yum.repos.d/nginx.repo
```
Define the official repository there:
```
[nginx]
name=nginx repo
baseurl=http://nginx.org/packages/rhel/7/$basearch/
gpgcheck=0
enabled=1
```
Now install with `yum`:
```
sudo yum install nginx
```

Check that it's working:
```bash
sudo service nginx start
# that'll start a server on the default port 80
# open roboao.caltech.edu to test and then stop the server
sudo service nginx stop
```

#### Installing `gunicorn` `WSGI`-server
Make sure (```which pip```) you're using `pip` from `anaconda`, then
```
pip install gunicorn
```

Try it from the directory with the `flask` app:
```
gunicorn -w 4 server_data_archive:app
```
This will start a server at port 8000 with 4 processes.

#### Configuring `nginx`

```
sudo vi /etc/nginx/nginx.conf
```

We should set up `nginx` to act in the reverse proxy mode to
redirect all incoming requests to `roboao.caltech.edu/archive` 
to the `gunicorn` server. This is what I have on the archive machine:
```
user  nginx;
worker_processes  1;

error_log  /var/log/nginx/error.log warn;
pid        /var/run/nginx.pid;


events {
    worker_connections  1024;
    accept_mutex off; # set to 'on' if nginx worker_processes > 1
    # 'use epoll;' to enable for Linux 2.6+
    # 'use kqueue;' to enable for FreeBSD, OSX
}


http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    sendfile        on;
    #tcp_nopush     on;

    #keepalive_timeout  65;

    #gzip  on;

    # default page:
    include /etc/nginx/conf.d/*.conf;

  upstream app_server {
    # fail_timeout=0 means we always retry an upstream even if it failed
    # to return a good HTTP response

    # for UNIX domain socket setups
    #server unix:/tmp/gunicorn.sock fail_timeout=0;

    # for a TCP configuration
    server 127.0.0.1:8000 fail_timeout=0;
  }

  server {
    # if no Host match, close the connection to prevent host spoofing
    listen 80 default_server;
    return 444;
  }

  server {
    # use 'listen 80 deferred;' for Linux
    # use 'listen 80 accept_filter=httpready;' for FreeBSD
    listen 80 deferred;
    client_max_body_size 4G;

    # set the correct host(s) for your site
    server_name roboao.caltech.edu www.roboao.caltech.edu;

    keepalive_timeout 5;

    location / {
        root   /usr/share/nginx/html;
        index  index.html index.htm;
    }

    # path for static files
    #root /data/roboao/dev/archiver;

    #location / {
    #    proxy_set_header X-Forward-For $proxy_add_x_forwarded_for;
    #    proxy_set_header Host $http_host;
    #    proxy_redirect off;
    #    rewrite /archive/(.*) /$1 break;
    #    if (!-f $request_filename) {
    #        proxy_pass http://app_server;
    #        break;
    #    }
    #}

    location /archive {
        #root /data/roboao/dev/archiver;
        #rewrite ^/(.*) /archive/$1 last;
        proxy_redirect off;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Scheme $scheme;
        proxy_set_header X-Script-Name /archive;
        if (!-f $request_filename) {
            proxy_pass http://app_server;
            break;
        }
#proxy_pass http://127.0.0.1:8000;
    }

    location /archive/static {
        alias  /data/roboao/dev/archiver/static;
        autoindex on;
    }
    location /archive/data {
        alias  /data/roboao/archive;
        autoindex on;
    }
  }

}
```

Make sure the `nginx` process has all the necessary `RHEL7` privileges
    to access/execute stuff. `ls -Z` is your friend here. 
    May have to do (one of) the following:
```
sudo audit2why -al
sudo semanage fcontext -a -t httpd_sys_content_t "/data/roboao/dev/archiver(/.*)?"
sudo setsebool -P httpd_can_network_connect 1
sudo semanage permissive -a httpd_t
```

Remember to `sudo service nginx restart` after each change.

#### Installing `supervisord` process manager

With the proper `pip`:
```
pip install supervisor
```

Now create a config file and edit it. I run `supervisord` as the 
`roboao` user, therefore I keep the config file in 
`/data/roboao/supervisord.conf`:
```
echo_supervisord_conf > /data/roboao/supervisord.conf
```

This is my current setup:
```
; Sample supervisor config file.
;
; For more information on the config file, please see:
; http://supervisord.org/configuration.html
;
; Notes:
;  - Shell expansion ("~" or "$HOME") is not supported.  Environment
;    variables can be expanded using this syntax: "%(ENV_HOME)s".
;  - Comments must have a leading space: "a=b ;comment" not "a=b;comment".

[unix_http_server]
file=/tmp/supervisor.sock   ; (the path to the socket file)
;chmod=0700                 ; socket file mode (default 0700)
;chown=nobody:nogroup       ; socket file uid:gid owner
;username=user              ; (default is no username (open server))
;password=123               ; (default is no password (open server))

;[inet_http_server]         ; inet (TCP) server disabled by default
;port=127.0.0.1:9001        ; (ip_address:port specifier, *:port for all iface)
;username=user              ; (default is no username (open server))
;password=123               ; (default is no password (open server))

[supervisord]
logfile=/tmp/supervisord.log ; (main log file;default $CWD/supervisord.log)
logfile_maxbytes=50MB        ; (max main logfile bytes b4 rotation;default 50MB)
logfile_backups=10           ; (num of main logfile rotation backups;default 10)
loglevel=info                ; (log level;default info; others: debug,warn,trace)
pidfile=/tmp/supervisord.pid ; (supervisord pidfile;default supervisord.pid)
nodaemon=false               ; (start in foreground if true;default false)
minfds=1024                  ; (min. avail startup file descriptors;default 1024)
minprocs=200                 ; (min. avail process descriptors;default 200)
;umask=022                   ; (process file creation umask;default 022)
;user=chrism                 ; (default is current user, required if root)
;identifier=supervisor       ; (supervisord identifier, default is 'supervisor')
;directory=/tmp              ; (default is not to cd during start)
;nocleanup=true              ; (don't clean up tempfiles at start;default false)
;childlogdir=/tmp            ; ('AUTO' child log dir, default $TEMP)
;environment=KEY="value"     ; (key value pairs to add to environment)
;strip_ansi=false            ; (strip ansi escape codes in logs; def. false)

; the below section must remain in the config file for RPC
; (supervisorctl/web interface) to work, additional interfaces may be
; added by defining them in separate rpcinterface: sections
[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///tmp/supervisor.sock ; use a unix:// URL  for a unix socket
;serverurl=http://127.0.0.1:9001 ; use an http:// url to specify an inet socket
;username=chris              ; should be same as http_username if set
;password=123                ; should be same as http_password if set
;prompt=mysupervisor         ; cmd line prompt (default "supervisor")
;history_file=~/.sc_history  ; use readline history if available

; The below sample program section shows all possible program subsection values,
; create one or more 'real' program: sections to be able to control them under
; supervisor.

;[program:theprogramname]
;command=/bin/cat              ; the program (relative uses PATH, can take args)
;process_name=%(program_name)s ; process_name expr (default %(program_name)s)
;numprocs=1                    ; number of processes copies to start (def 1)
;directory=/tmp                ; directory to cwd to before exec (def no cwd)
;umask=022                     ; umask for process (default None)
;priority=999                  ; the relative start priority (default 999)
;autostart=true                ; start at supervisord start (default: true)
;startsecs=1                   ; # of secs prog must stay up to be running (def. 1)
;startretries=3                ; max # of serial start failures when starting (default 3)
;autorestart=unexpected        ; when to restart if exited after running (def: unexpected)
;exitcodes=0,2                 ; 'expected' exit codes used with autorestart (default 0,2)
;stopsignal=QUIT               ; signal used to kill process (default TERM)
;stopwaitsecs=10               ; max num secs to wait b4 SIGKILL (default 10)
;stopasgroup=false             ; send stop signal to the UNIX process group (default false)
;killasgroup=false             ; SIGKILL the UNIX process group (def false)
;user=chrism                   ; setuid to this UNIX account to run the program
;redirect_stderr=true          ; redirect proc stderr to stdout (default false)
;stdout_logfile=/a/path        ; stdout log path, NONE for none; default AUTO
;stdout_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
;stdout_logfile_backups=10     ; # of stdout logfile backups (default 10)
;stdout_capture_maxbytes=1MB   ; number of bytes in 'capturemode' (default 0)
;stdout_events_enabled=false   ; emit events on stdout writes (default false)
;stderr_logfile=/a/path        ; stderr log path, NONE for none; default AUTO
;stderr_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
;stderr_logfile_backups=10     ; # of stderr logfile backups (default 10)
;stderr_capture_maxbytes=1MB   ; number of bytes in 'capturemode' (default 0)
;stderr_events_enabled=false   ; emit events on stderr writes (default false)
;environment=A="1",B="2"       ; process environment additions (def no adds)
;serverurl=AUTO                ; override serverurl computation (childutils)

; The below sample eventlistener section shows all possible
; eventlistener subsection values, create one or more 'real'
; eventlistener: sections to be able to handle event notifications
; sent by supervisor.

;[eventlistener:theeventlistenername]
;command=/bin/eventlistener    ; the program (relative uses PATH, can take args)
;process_name=%(program_name)s ; process_name expr (default %(program_name)s)
;numprocs=1                    ; number of processes copies to start (def 1)
;events=EVENT                  ; event notif. types to subscribe to (req'd)
;buffer_size=10                ; event buffer queue size (default 10)
;directory=/tmp                ; directory to cwd to before exec (def no cwd)
;umask=022                     ; umask for process (default None)
;priority=-1                   ; the relative start priority (default -1)
;autostart=true                ; start at supervisord start (default: true)
;startsecs=1                   ; # of secs prog must stay up to be running (def. 1)
;startretries=3                ; max # of serial start failures when starting (default 3)
;autorestart=unexpected        ; autorestart if exited after running (def: unexpected)
;exitcodes=0,2                 ; 'expected' exit codes used with autorestart (default 0,2)
;stopsignal=QUIT               ; signal used to kill process (default TERM)
;stopwaitsecs=10               ; max num secs to wait b4 SIGKILL (default 10)
;stopasgroup=false             ; send stop signal to the UNIX process group (default false)
;killasgroup=false             ; SIGKILL the UNIX process group (def false)
;user=chrism                   ; setuid to this UNIX account to run the program
;redirect_stderr=false         ; redirect_stderr=true is not allowed for eventlisteners
;stdout_logfile=/a/path        ; stdout log path, NONE for none; default AUTO
;stdout_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
;stdout_logfile_backups=10     ; # of stdout logfile backups (default 10)
;stdout_events_enabled=false   ; emit events on stdout writes (default false)
;stderr_logfile=/a/path        ; stderr log path, NONE for none; default AUTO
;stderr_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
;stderr_logfile_backups=10     ; # of stderr logfile backups (default 10)
;stderr_events_enabled=false   ; emit events on stderr writes (default false)
;environment=A="1",B="2"       ; process environment additions
;serverurl=AUTO                ; override serverurl computation (childutils)

; The below sample group section shows all possible group values,
; create one or more 'real' group: sections to create "heterogeneous"
; process groups.

;[group:thegroupname]
;programs=progname1,progname2  ; each refers to 'x' in [program:x] definitions
;priority=999                  ; the relative start priority (default 999)

; The [include] section can just contain the "files" setting.  This
; setting can list multiple files (separated by whitespace or
; newlines).  It can also contain wildcards.  The filenames are
; interpreted as relative to this file.  Included files *cannot*
; include files themselves.

;[include]
;files = relative/directory/*.ini

[program:roboao-archive]
command = /data/roboao/anaconda3/bin/gunicorn -w 4 server_data_archive:app
directory = /data/roboao/dev/archiver
user = roboao
stdout_logfile = /data/roboao/dev/archiver/logs/gunicorn_stdout.log
stderr_logfile = /data/roboao/dev/archiver/logs/gunicorn_stderr.log
redirect_stderr = True
environment = PRODUCTION=1
```

Finally, start the `supervisor` daemon process, restart `nginx`,
and you're production!
```
supervisord -c /data/roboao/supervisord.conf
```

#### Make sure to start everything on system boot

First, enable starting nginx service 
(assuming your OS is using systemd, like RHEL) upon system boot:
```
sudo systemctl enable nginx.service
```

Set up cronjobs to start MongoDB (if manually installed) and supervisord:
```
>>> crontab -e
```

This is what my current cron set-up looks like:
```
#MAILTO=""
MAILTO=duev@caltech.edu
# start mongodb in background mode at start-up
#@reboot /data/roboao/mongodb/bin/mongod -f /data/roboao/mongodb/mongod.conf >/dev/null 2>&1
@reboot /data/roboao/mongodb/bin/mongod -f /data/roboao/mongodb/mongod.conf
# start public and data archive servers
@reboot /data/roboao/anaconda2/bin/python /data/roboao/anaconda2/bin/supervisord -c /data/roboao/supervisord.master.conf
```

#### Notes

If you had an apache server running on the machine, you'd want to 
prevent starting it on reboot:
```
sudo systemctl disable httpd.service
```

