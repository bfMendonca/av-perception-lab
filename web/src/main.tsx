import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createClient } from "./api/client";
import { App } from "./App";
import "./ui/theme.css";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");

createRoot(root).render(
  <StrictMode>
    <App client={createClient()} persistDraft />
  </StrictMode>,
);
