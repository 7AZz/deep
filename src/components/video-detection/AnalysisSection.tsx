import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, CheckCircle2 } from "lucide-react";
import VideoPlayer from "./VideoPlayer";

interface AnalysisSectionProps {
  videoUrl?: string;
  isAnalyzing?: boolean;
  confidenceScore?: number;
  detectionResult?: "real" | "fake" | "REAL" | "FAKE" | null;
  onVideoTimeUpdate?: (time: number) => void;
}

const AnalysisSection = ({
  videoUrl = "https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
  isAnalyzing = false,
  confidenceScore = 85,
  detectionResult = "fake",
  onVideoTimeUpdate = () => {},
}: AnalysisSectionProps) => {
  const [currentFrame, setCurrentFrame] = useState(0);

  const handleTimeUpdate = (time: number) => {
    setCurrentFrame(Math.floor(time * 30)); // Assuming 30fps
    onVideoTimeUpdate(time);
  };

  return (
    <div className="w-full space-y-6 bg-gray-50 p-6 rounded-lg">
      <div className="flex flex-col md:flex-row gap-6">
        <div className="flex-1">
          <VideoPlayer src={videoUrl} onTimeUpdate={handleTimeUpdate} />
          {detectionResult && (
            <div
              className={`mt-4 p-4 rounded-lg ${detectionResult === "REAL" ? "bg-green-100" : "bg-red-100"}`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold mb-2">
                    Analysis Results
                  </h3>
                  <div className="flex items-center space-x-2">
                    <Badge
                      variant={
                        detectionResult === "FAKE" ? "destructive" : "success"
                      }
                      className="flex items-center gap-1"
                    >
                      {detectionResult === "FAKE" ? (
                        <AlertCircle className="h-4 w-4" />
                      ) : (
                        <CheckCircle2 className="h-4 w-4" />
                      )}
                      {detectionResult === "FAKE"
                        ? "Likely Deepfake"
                        : "Likely Authentic"}
                    </Badge>
                  </div>
                </div>
                {confidenceScore && (
                  <div className="text-right">
                    <span className="text-sm text-gray-600">Confidence</span>
                    <div className="text-2xl font-bold">
                      {confidenceScore.toFixed(1)}%
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <Card className="flex-1 p-6 bg-white">
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">Analysis Results</h3>
              <div className="flex items-center space-x-2">
                {detectionResult === "fake" ? (
                  <Badge
                    variant="destructive"
                    className="flex items-center gap-1"
                  >
                    <AlertCircle className="h-4 w-4" />
                    Likely Deepfake
                  </Badge>
                ) : (
                  <Badge variant="success" className="flex items-center gap-1">
                    <CheckCircle2 className="h-4 w-4" />
                    Likely Authentic
                  </Badge>
                )}
              </div>
            </div>

            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm text-gray-600">Confidence Score</span>
                <span className="font-medium">{confidenceScore}%</span>
              </div>
              <Progress value={confidenceScore} className="h-2" />
            </div>

            <div>
              <h4 className="text-sm font-medium mb-2">Analysis Details</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Current Frame</span>
                  <span>{currentFrame}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Processing Status</span>
                  <span>{isAnalyzing ? "Analyzing..." : "Complete"}</span>
                </div>
              </div>
            </div>

            {isAnalyzing && (
              <div className="text-center text-sm text-gray-600">
                <Progress value={33} className="h-2 mb-2" />
                Processing video frames...
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};

export default AnalysisSection;
