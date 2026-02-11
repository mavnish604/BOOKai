import React from 'react';

interface ChatInputProps {
    query: string;
    setQuery: (query: string) => void;
    handleSubmit: (e: React.FormEvent) => void;
    isLoading: boolean;
}

export default function ChatInput({ query, setQuery, handleSubmit, isLoading }: ChatInputProps) {
    return (
        <div className="bg-white border-t border-gray-200 p-4">
            <form
                className="max-w-4xl mx-auto relative flex items-center gap-2"
                onSubmit={handleSubmit}
            >
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask for a book recommendation based on your mood or interests..."
                    disabled={isLoading}
                    className="w-full bg-gray-50 text-gray-900 rounded-full border border-gray-300 pl-5 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all placeholder-gray-400 disabled:bg-gray-100 disabled:cursor-not-allowed"
                />

                <button
                    type="submit"
                    disabled={isLoading || !query.trim()}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors shadow-md flex items-center justify-center w-9 h-9"
                    aria-label="Send message"
                >
                    {isLoading ? (
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4 ml-0.5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
                        </svg>
                    )}
                </button>
            </form>
        </div>
    );
}
