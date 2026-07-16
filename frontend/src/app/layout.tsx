import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "InsideAI — Real-Time Transformer Visualization",
  description:
    "Watch a real transformer process your prompt: tokenization, embeddings, attention, MLP activations, logits and sampling — streamed live from the model.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
