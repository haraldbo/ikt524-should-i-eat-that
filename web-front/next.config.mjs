/** @type {import('next').NextConfig} */

// next.config.js
const API_URL = "http://127.0.0.1:5000"; // e.g., 'https://api.example.com'

/** @type {import('next').NextConfig} */
const nextConfig = {
    async rewrites() {
        return [
            {
                source: '/api/:path*', // Incoming path
                destination: `${API_URL}/api/:path*`, // Destination path
            },
        ];
    },
};

export default nextConfig;