# Use Node.js image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package.json and install deps
COPY package.json package-lock.json ./
RUN npm install --frozen-lockfile

# Copy the rest of the frontend
COPY . .

# Build Next.js app
RUN npm run build

# Expose frontend port
EXPOSE 3000

# Run Next.js
CMD ["npm", "start"]
