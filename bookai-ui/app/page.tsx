"use client"; // Critical for Next.js to run in the browser

import React, { useState, useRef, useEffect } from 'react';

// API Base URL (your FastAPI server)
const API_URL = 'http://127.0.0.1:8000';

// --- BookCard Component (with Cover, Score, and Summary Fetching) ---
interface RecommendationProps {
  title: string;
  explanation: string;
  score: number; // New score field
}

function BookCard({ title, explanation, score }: RecommendationProps) {
  const [summary, setSummary] = useState<string | null>(null);
  const [coverImage, setCoverImage] = useState<string | null>(null);
  const [isSummaryLoading, setIsSummaryLoading] = useState(false);
  const [isLogVisible, setIsLogVisible] = useState(false); // Controls log panel visibility

  // --- 1. Fetch cover image when component loads ---
  useEffect(() => {
    async function fetchCover() {
      try {
        // NOTE: This key is visible in the browser. For a real app, 
        // you would use a backend proxy to hide it.
        const apiKey = "lun"; 
        
        const response = await fetch(
          `https://www.googleapis.com/books/v1/volumes?q=intitle:${encodeURIComponent(title)}&key=${apiKey}`
        );
        const data = await response.json();
        const cover = data.items?.[0]?.volumeInfo?.imageLinks?.thumbnail;
        if (cover) {
          setCoverImage(cover);
        }
      } catch (error) {
        console.error("Error fetching book cover:", error);
      }
    }
    fetchCover();
  }, [title]);

  // --- 2. Summary Fetching Logic (Non-Streaming for stability) ---
  const fetchSummary = async () => {
    if (isSummaryLoading || summary) {
      setIsLogVisible(true); // Toggle log view if summary already exists
      return; 
    }

    setIsSummaryLoading(true);
    setIsLogVisible(true); // Open log panel
    try {
      // Since we reverted the backend, this is a standard POST for the final summary
      const response = await fetch(`${API_URL}/summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ book_title: title }),
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setSummary(data.summary);

    } catch (error) {
      console.error('Error fetching summary:', error);
      setSummary('Sorry, I couldn\'t get a summary for this book.');
    } finally {
      setIsSummaryLoading(false);
    }
  };

  return (
    <div className="book-card">
      {/* Cover */}
      <div className="book-cover-container">
        {coverImage ? (
          <img src={coverImage} alt={`Cover of ${title}`} className="book-cover" />
        ) : (
          <div className="book-cover-placeholder">?</div>
        )}
      </div>

      {/* Info */}
      <div className="book-info">
        <h4>{title}</h4>
        
        {/* --- NEW: Display the Score --- */}
        <p className="score">
            Chance of Liking: <strong>{Math.round(score * 100)}%</strong>
        </p>

        <p className="explanation">
          <strong>Why?</strong> {explanation}
        </p>
        
        <button onClick={fetchSummary} disabled={isSummaryLoading}>
          {isSummaryLoading ? 'Getting Summary...' : (summary ? 'View Summary' : 'Get Summary')}
        </button>

        {/* --- Collapsible Summary Panel --- */}
        {(summary || isSummaryLoading) && (
          <details className="summary-details" open={isLogVisible} onToggle={(e) => setIsLogVisible((e.target as HTMLDetailsElement).open)}>
            <summary>
              {isSummaryLoading ? "Agent is working..." : "Summary (Click to hide)"}
            </summary>
            
            <div className="summary">
                {isSummaryLoading ? (
                    <div className="summary-loading">Fetching detailed information...</div>
                ) : (
                    <p>{summary}</p>
                )}
            </div>
            
          </details>
        )}
      </div>
    </div>
  );
}

// --- Main App Component (Your Page) ---
export default function Home() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([
    {
      sender: 'bot',
      text: 'Hello! I am BOOKai. Ask me for a recommendation',
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
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    const userMessage = { sender: 'user', text: query, recommendations: [] };
    setMessages((prev) => [...prev, userMessage]);
    setQuery('');

    try {
      // 1. Call the /recommend endpoint
      const response = await fetch(`${API_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();

      // 2. Add the bot's response to the chat
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
        text: 'Sorry, I ran into an error. Please check my console (F12) and my server logs.',
        recommendations: [],
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="chat-window" ref={chatWindowRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <p>{msg.text}</p>
            {msg.recommendations && msg.recommendations.length > 0 && (
              <div className="recommendations">
                {msg.recommendations.map((rec, i) => (
                  <BookCard
                    key={i}
                    title={rec.title}
                    explanation={rec.explanation}
                    score={rec.score} // Pass the new score prop
                  />
                ))}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="message bot">
            <p>BOOKai is thinking...</p>
          </div>
        )}
      </div>
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask for a recommendation..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          Send
        </button>
      </form>
    </div>
  );
}