# ---- Stage 1: build the React app ----
FROM node:22-alpine AS build
WORKDIR /app

# Install deps first (cached unless package files change)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build -> /app/dist
COPY . .
ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

# ---- Stage 2: serve with nginx ----
FROM nginx:1.27-alpine
# Copied as a *template*: at startup the nginx image replaces ${BACKEND_URL} with the
# env var's value and writes the result to /etc/nginx/conf.d/default.conf.
COPY nginx.conf /etc/nginx/templates/default.conf.template
# Default = docker-compose backend. Azure overrides it with --env-vars BACKEND_URL=...
ENV BACKEND_URL=http://backend:8000
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
