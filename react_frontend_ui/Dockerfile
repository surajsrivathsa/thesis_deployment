FROM node:19-alpine
RUN apk update && apk add git

RUN mkdir /app
WORKDIR /app
# # COPY package*.json /app

# # install latest npm and clean cache
# # RUN npm install --location=global npm@8.11.0


# # RUN npm i --silent
# # RUN npm cache clean --force
# # RUN npm i --location=global react-scripts@5.0.1

COPY . /app


# # CMD ["npm", "start"]

RUN npm install --legacy-peer-deps
EXPOSE 3000

# Set the environment variable for HTTPS and the SSL certificate paths
ENV HTTPS=true
ENV SSL_CERTIFICATE=/app/cert.pem
ENV SSL_KEY=/app/key.pem

# Start your React application
CMD ["npm", "start"]




