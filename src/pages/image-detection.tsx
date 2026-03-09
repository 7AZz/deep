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

export default function ImageDetection() {
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<{
    result?: "REAL" | "FAKE";
    confidence?: number;
    face_detected?: boolean;
    image_size?: [number, number, number];
  } | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const { user } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleAnalyze = async () => {
    if (!user) {
      toast({
        title: "Sign in required",
        description: "Please sign in to analyze images",
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
      const response = await api.analyzeImage(file, (progress) => {
        setProgress(progress);
      });

      setResult(response);

      // Show success toast
      toast({
        title: `Image Analysis Complete`,
        description: `Result: ${response.result} (${response.confidence.toFixed(1)}% confidence)`,
        variant: response.result === "FAKE" ? "destructive" : "default",
      });
    } catch (error) {
      console.error("Analysis failed:", error);
      toast({
        title: "Analysis Failed",
        description:
          error instanceof Error ? error.message : "Failed to analyze image",
        variant: "destructive",
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setAnalyzing(false);
    setProgress(0);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <BackButton />
      <div className="container max-w-3xl pt-32 pb-20">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">Image Deepfake Detection</h1>
          <p className="text-muted-foreground">
            Upload an image to analyze it for potential AI manipulation
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
                acceptedTypes={["image"]}
                onFileSelect={setFile}
                uploading={analyzing}
                progress={progress}
                maxSize={20 * 1024 * 1024} // 20MB
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
                <h3 className="text-lg font-semibold">Analyzing Image</h3>
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
                          AI Generated Image
                        </>
                      ) : (
                        <>
                          <CheckCircle2 className="w-4 h-4 mr-1" />
                          Natural Image
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

                  {/* Additional Details */}
                  <div className="grid grid-cols-2 gap-4 pt-4">
                    <div className="space-y-2">
                      <div className="text-sm text-muted-foreground">
                        Face Detection
                      </div>
                      <div className="text-2xl font-bold">
                        {result.face_detected ? "Detected" : "Not Found"}
                      </div>
                    </div>
                    {result.image_size && (
                      <div className="space-y-2">
                        <div className="text-sm text-muted-foreground">
                          Image Size
                        </div>
                        <div className="text-sm">
                          {result.image_size[1]}x{result.image_size[0]} pixels
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
