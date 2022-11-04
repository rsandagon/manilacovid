FROM nginx:stable-alpine
COPY app/default.conf /etc/nginx/conf.d
ADD app /usr/share/nginx/html/
EXPOSE 8080
EXPOSE 80
CMD ["/bin/sh", "-c", "exec nginx -g 'daemon off;';"]
WORKDIR /usr/share/nginx/html