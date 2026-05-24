import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AppShell } from "@/app";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root not found");

createRoot(root).render(
  <StrictMode>
    <AppShell />
  </StrictMode>,
);
