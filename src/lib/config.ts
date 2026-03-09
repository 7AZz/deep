const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export const config = {
  apiUrl,
  endpoints: {
    video: `${apiUrl}/ml_app/api/analyze/`,
    image: `${apiUrl}/ml_app/api/analyze-image/`,
    audio: `${apiUrl}/ml_app/api/analyze-audio/`,
  },
};
