import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Upload, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { FileUploadZone } from "@/components/upload/FileUploadZone";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import Navbar from "@/components/navbar";
import { BackButton } from "@/components/ui/back-button";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/lib/use-toast";
import { useNavigate } from "react-router-dom";

export default function VideoDetection() {
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<{
    result?: "REAL" | "FAKE";
    confidence?: number;
    frame_predictions?: Array<[boolean, number]>;
    faces_detected?: boolean[];
    total_frames?: number;
    frames_with_faces?: number;
  } | null>(null);

  const { user } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleAnalyze = async () => {
    if (!user) {
      toast({
        title: "Sign in required",
        description: "Please sign in to analyze videos",
        variant: "default",
      });
      navigate("/sign-in");
      return;
    }
    if (!file) return;

    try {
      setAnalyzing(true);
      setProgress(0);
      setResult(null);

      // Send to backend
      const response = await api.analyzeVideo(file, (progress) => {
        setProgress(progress);
      });

      setResult(response);

      // Show success toast
      toast({
        title: `Video Analysis Complete`,
        description: `Result: ${response.result} (${response.confidence.toFixed(1)}% confidence)`,
        variant: response.result === "FAKE" ? "destructive" : "default",
      });
    } catch (error) {
      console.error("Analysis failed:", error);
      toast({
        title: "Analysis Failed",
        description:
          error instanceof Error ? error.message : "Failed to analyze video",
        variant: "destructive",
      });
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <BackButton />
      <div className="container max-w-3xl pt-32 pb-20">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">Video Deepfake Detection</h1>
          <p className="text-muted-foreground">
            Upload a video to analyze it for potential deepfake manipulation
          </p>
        </div>

        <div className="space-y-6">
          {/* Upload Section */}
          <Card className="p-8">
            <div className="relative">
              {file && (
                <Button
                  variant="outline"
                  size="icon"
                  className="absolute top-2 right-2 z-10"
                  onClick={(e) => {
                    e.stopPropagation();
                    window.location.reload();
                  }}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="w-4 h-4"
                  >
                    <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.85.99 6.57 2.57" />
                    <path d="M21 3v9h-9" />
                  </svg>
                </Button>
              )}
              <FileUploadZone
                acceptedTypes={["video"]}
                onFileSelect={setFile}
                uploading={analyzing}
                progress={progress}
                maxSize={100 * 1024 * 1024} // 100MB
              />

              {file && !analyzing && !result && (
                <div className="mt-6 flex justify-center">
                  <Button onClick={handleAnalyze} size="lg">
                    Start Analysis
                  </Button>
                </div>
              )}
            </div>
          </Card>

          {/* Analysis Progress */}
          {analyzing && (
            <Card className="p-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Analyzing Video</h3>
                <Progress value={progress} />
                <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Processing... {progress}%
                </div>
              </div>
            </Card>
          )}

          {/* Results */}
          {result && (
            <div className="space-y-6">
              <Card className="p-6">
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Analysis Results</h3>
                    <Badge
                      variant={
                        result.result === "FAKE" ? "destructive" : "default"
                      }
                      className="text-sm"
                    >
                      {result.result === "FAKE" ? (
                        <>
                          <AlertCircle className="w-4 h-4 mr-1" />
                          Likely Deepfake
                        </>
                      ) : (
                        <>
                          <CheckCircle2 className="w-4 h-4 mr-1" />
                          Likely Authentic
                        </>
                      )}
                    </Badge>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">
                        Confidence Score
                      </span>
                      <span>{result.confidence?.toFixed(1)}%</span>
                    </div>
                    <Progress
                      value={result.confidence}
                      className={
                        result.result === "FAKE" ? "bg-destructive/20" : ""
                      }
                      indicatorClassName={
                        result.result === "FAKE" ? "bg-destructive" : ""
                      }
                    />
                  </div>

                  {/* Additional Analysis Details */}
                  <div className="grid grid-cols-2 gap-4 pt-4">
                    <div className="space-y-2">
                      <div className="text-sm text-muted-foreground">
                        Total Frames
                      </div>
                      <div className="text-2xl font-bold">
                        {result.total_frames}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm text-muted-foreground">
                        Faces Detected
                      </div>
                      <div className="text-2xl font-bold">
                        {result.frames_with_faces}
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Frame Analysis Visualization */}
              {result.frame_predictions && (
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Frame Analysis</h3>
                  <div className="h-20 relative border rounded-md overflow-hidden">
                    {result.frame_predictions.map((prediction, i) => (
                      <div
                        key={i}
                        className={`absolute w-1 h-full transition-colors ${
                          prediction[0] ? "bg-destructive" : "bg-primary"
                        }`}
                        style={{
                          left: `${(i / result.frame_predictions.length) * 100}%`,
                          opacity: prediction[1],
                        }}
                        title={`Frame ${i}: ${(prediction[1] * 100).toFixed(1)}% confidence`}
                      />
                    ))}
                  </div>
                  <div className="flex justify-between mt-2 text-sm text-muted-foreground">
                    <span>Start</span>
                    <span>End</span>
                  </div>
                </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
