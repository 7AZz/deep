import { Routes, Route, useRoutes } from "react-router-dom";
import Team from "./pages/team";
import Home from "./components/home";
import VideoDetection from "./pages/video-detection";
import ImageDetection from "./pages/image-detection";
import AudioDetection from "./pages/audio-detection";
import SignIn from "./components/auth/sign-in";
import Register from "./components/auth/register";
import Dashboard from "./components/dashboard";
import { AuthProvider } from "./contexts/AuthContext";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { Toaster } from "./components/ui/toaster";
import routes from "tempo-routes";

function App() {
  return (
    <AuthProvider>
      {/* Tempo routes */}
      {import.meta.env.VITE_TEMPO && useRoutes(routes)}

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/sign-in" element={<SignIn />} />
        <Route path="/register" element={<Register />} />
        <Route path="/team" element={<Team />} />

        {/* Detection Routes */}
        <Route path="/video-detection" element={<VideoDetection />} />
        <Route path="/image-detection" element={<ImageDetection />} />
        <Route path="/audio-detection" element={<AudioDetection />} />

        {/* Protected Routes */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />

        {/* Add this before any catchall routes */}
        {import.meta.env.VITE_TEMPO && <Route path="/tempobook/*" />}
      </Routes>
      <Toaster />
    </AuthProvider>
  );
}

export default App;
