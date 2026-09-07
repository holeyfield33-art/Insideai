import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "InsideAI — Model Observatory",
  description:
    "Watch a real transformer process your prompt: tokenization, embeddings, attention, MLP activations, logits and sampling — streamed live from the model.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      {/* suppressHydrationWarning: browser extensions (ColorZilla, Grammarly…)
          inject attributes into <body> before React hydrates. */}
      <body className="font-sans antialiased" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
