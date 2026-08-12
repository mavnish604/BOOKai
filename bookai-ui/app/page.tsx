"use client";

import React, { useState, useRef, useEffect } from 'react';
import Header from './components/Header';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL;
export default function Home() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([
    {
      sender: 'bot',
      text: 'Hello! I am BOOKai. Tell me what kind of books you like, and I’ll find the perfect match for you.',
      recommendations: [],
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const chatWindowRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages, isLoading]); // also scroll when loading state changes (e.g. typing indicator)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    const userMessage = { sender: 'user', text: query, recommendations: [] };
    setMessages((prev) => [...prev, userMessage]);
    setQuery('');

    try {
      const response = await fetch(`${API_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();

      const botMessage = {
        sender: 'bot',
        text: data.response,
        recommendations: data.recommendations || [],
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      const errorMessage = {
        sender: 'bot',
        text: 'Sorry, I ran into an error connecting to the server. Please check if the backend is running.',
        recommendations: [],
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[100dvh] bg-gray-50 font-sans text-gray-900">
      <Header />

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 sm:p-6 scroll-smooth" ref={chatWindowRef}>
        <div className="max-w-4xl mx-auto">
          {messages.map((msg, index) => (
            <ChatMessage
              key={index}
              sender={msg.sender}
              text={msg.text}
              recommendations={msg.recommendations}
            />
          ))}

          {isLoading && (
            <div className="flex w-full justify-start mb-6 animate-pulse">
              <div className="flex max-w-[80%] gap-3 flex-row">
                <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white text-sm font-bold shadow-sm">B</div>
                <div className="px-4 py-3 bg-white border border-gray-100 rounded-2xl rounded-tl-none shadow-sm text-gray-500 italic text-[15px]">
                  Thinking...
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      <ChatInput
        query={query}
        setQuery={setQuery}
        handleSubmit={handleSubmit}
        isLoading={isLoading}
      />
    </div>
  );
}
