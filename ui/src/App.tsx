import { Navigate, Route, Routes } from "react-router-dom";
import TopBar from "./components/TopBar";
import { ToastProvider } from "./components/Toast";
import Live from "./pages/Live";
import Review from "./pages/Review";
import Results from "./pages/Results";
import Provenance from "./pages/Provenance";

export default function App() {
  return (
    <ToastProvider>
      <div className="min-h-full flex flex-col">
        <TopBar />
        <main className="flex-1 w-full max-w-[1440px] mx-auto px-5 md:px-8 py-6">
          <Routes>
            <Route path="/" element={<Live />} />
            <Route path="/review" element={<Review />} />
            <Route path="/results" element={<Results />} />
            <Route path="/provenance" element={<Provenance />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </ToastProvider>
  );
}
