"use client";

import React, { useState, useEffect } from 'react';

interface BookCardProps {
    title: string;
    explanation: string;
    score: number;
}

export default function BookCard({ title, explanation, score }: BookCardProps) {
    const [summary, setSummary] = useState<string | null>(null);
    const [coverImage, setCoverImage] = useState<string | null>(null);
    const [isSummaryLoading, setIsSummaryLoading] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);

    // API Base URL (hardcoded for now as in original)
    const API_URL = 'http://127.0.0.1:8000';

    useEffect(() => {
        async function fetchCover() {
            try {
                const apiKey = "AIzaSyA2_7uddDbJ5SFaVPsVoRCm1-yI_zrBf48";
                const response = await fetch(
                    `https://www.googleapis.com/books/v1/volumes?q=intitle:${encodeURIComponent(title)}&key=${apiKey}&printType=books`
                );
                const data = await response.json();
                const cover = data.items?.[0]?.volumeInfo?.imageLinks?.thumbnail;
                if (cover) {
                    setCoverImage(cover.replace('http:', 'https:'));
                }
            } catch (error) {
                console.error("Error fetching book cover:", error);
            }
        }
        fetchCover();
    }, [title]);

    const fetchSummary = async () => {
        if (summary) {
            setIsExpanded(!isExpanded);
            return;
        }

        setIsSummaryLoading(true);
        setIsExpanded(true); // Expand to show loading state

        try {
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

    const percentage = Math.round(score * 100);

    // Dynamic color for score
    const scoreColor = percentage >= 80 ? 'text-green-600 bg-green-50' :
        percentage >= 60 ? 'text-yellow-600 bg-yellow-50' :
            'text-red-600 bg-red-50';

    return (
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow duration-300 flex flex-col sm:flex-row gap-4 p-4 mt-3 max-w-2xl w-full">
            {/* Cover Image */}
            <div className="flex-shrink-0 w-24 h-36 bg-gray-100 rounded-md overflow-hidden self-center sm:self-start relative">
                {coverImage ? (
                    <img
                        src={coverImage}
                        alt={`Cover of ${title}`}
                        className="w-full h-full object-cover"
                        referrerPolicy="no-referrer"
                        onError={(e) => {
                            e.currentTarget.style.display = 'none';
                            console.error(`Failed to load image: ${coverImage}`);
                        }}
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-300 text-4xl font-serif">?</div>
                )}
            </div>

            {/* Content */}
            <div className="flex-grow flex flex-col justify-between">
                <div>
                    <div className="flex justify-between items-start mb-2">
                        <h3 className="text-lg font-bold text-gray-900 leading-tight">{title}</h3>
                        {/* Score Badge */}
                        <div className={`text-xs font-bold px-2 py-1 rounded-full ${scoreColor} ml-2 flex-shrink-0`}>
                            {percentage}% Match
                        </div>
                    </div>

                    <p className="text-sm text-gray-600 mb-3 leading-relaxed">
                        <span className="font-medium text-gray-900">Why?</span> {explanation}
                    </p>
                </div>

                <div>
                    <button
                        onClick={fetchSummary}
                        disabled={isSummaryLoading}
                        className="text-sm font-medium text-blue-600 hover:text-blue-800 hover:underline focus:outline-none disabled:opacity-50 transition-colors"
                    >
                        {isSummaryLoading ? 'Analyzing...' : (summary && !isExpanded ? 'Show Summary' : (summary && isExpanded ? 'Hide Summary' : 'Get Summary'))}
                    </button>

                    {/* Animated Summary Section */}
                    <div className={`overflow-hidden transition-all duration-300 ease-in-out ${isExpanded ? 'max-h-96 opacity-100 mt-3' : 'max-h-0 opacity-0'}`}>
                        <div className="bg-gray-50 rounded-lg p-3 text-sm text-gray-700 leading-relaxed border border-gray-100">
                            {isSummaryLoading ? (
                                <div className="flex items-center gap-2 text-gray-500">
                                    <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                                    Writing summary...
                                </div>
                            ) : (
                                summary
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
