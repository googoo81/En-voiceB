"use client";
import { useState } from "react";

export default function Home() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");

  const sendMessage = async () => {
    const res = await fetch("http://localhost:5001/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await res.json();
    setResponse(data.response);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-4">
      <input
        className="border rounded p-2"
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="메시지를 입력하세요"
      />
      <button
        className="px-4 py-2 bg-blue-500 text-white rounded"
        onClick={sendMessage}
      >
        전송
      </button>
      {response && (
        <div className="mt-4 p-4 border rounded bg-gray-100">{response}</div>
      )}
    </div>
  );
}
