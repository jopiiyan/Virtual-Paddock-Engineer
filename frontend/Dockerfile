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
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
