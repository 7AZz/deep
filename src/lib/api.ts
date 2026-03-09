import axios from "axios";
import { config } from "./config";

export const api = {
  async analyzeVideo(file: File, onProgress?: (progress: number) => void) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(config.endpoints.video, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total,
            );
            onProgress(percentCompleted);
          }
        },
      });

      const data = response.data;

      return {
        result: data.result as "REAL" | "FAKE",
        confidence: parseFloat(data.confidence.toFixed(2)),
        video_url: data.video_url,
        filename: data.filename,
        frame_predictions: data.frame_predictions,
        faces_detected: data.faces_detected,
        total_frames: data.total_frames,
        frames_with_faces: data.frames_with_faces,
      };
    } catch (error) {
      console.error("Error analyzing video:", error);
      throw new Error(
        error instanceof Error ? error.message : "Failed to analyze video",
      );
    }
  },

  async analyzeImage(file: File, onProgress?: (progress: number) => void) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(config.endpoints.image, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total,
            );
            onProgress(percentCompleted);
          }
        },
      });

      const data = response.data;

      return {
        result: data.result as "REAL" | "FAKE",
        confidence: parseFloat(data.confidence.toFixed(2)),
        face_detected: data.face_detected,
        image_size: data.image_size,
      };
    } catch (error) {
      console.error("Error analyzing image:", error);
      throw new Error(
        error instanceof Error ? error.message : "Failed to analyze image",
      );
    }
  },

  async analyzeAudio(file: File, onProgress?: (progress: number) => void) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(config.endpoints.audio, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total,
            );
            onProgress(percentCompleted);
          }
        },
      });

      const data = response.data;

      return {
        result: data.result as "REAL" | "FAKE",
        confidence: parseFloat(data.confidence.toFixed(2)),
      };
    } catch (error) {
      console.error("Error analyzing audio:", error);
      throw new Error(
        error instanceof Error ? error.message : "Failed to analyze audio",
      );
    }
  },
};
