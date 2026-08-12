import { NextRequest, NextResponse } from 'next/server';

const GOOGLE_BOOKS_API_URL = 'https://www.googleapis.com/books/v1/volumes';

export async function GET(request: NextRequest) {
    const title = request.nextUrl.searchParams.get('title')?.trim();

    if (!title) {
        return NextResponse.json({ error: 'A book title is required.' }, { status: 400 });
    }

    const apiKey = process.env.GOOGLE_BOOKS_API_KEY;
    if (!apiKey) {
        console.error('GOOGLE_BOOKS_API_KEY is not configured.');
        return NextResponse.json({ error: 'Book cover service is not configured.' }, { status: 503 });
    }

    const searchParams = new URLSearchParams({
        q: `intitle:${title}`,
        key: apiKey,
        printType: 'books',
        maxResults: '1',
    });

    try {
        const response = await fetch(`${GOOGLE_BOOKS_API_URL}?${searchParams}`, {
            next: { revalidate: 86400 },
        });

        if (!response.ok) {
            console.error(`Google Books request failed with status ${response.status}.`);
            return NextResponse.json({ error: 'Unable to retrieve book cover.' }, { status: 502 });
        }

        const data = await response.json();
        const cover = data.items?.[0]?.volumeInfo?.imageLinks?.thumbnail;

        return NextResponse.json({
            cover: typeof cover === 'string' ? cover.replace('http:', 'https:') : null,
        });
    } catch (error) {
        console.error('Error retrieving book cover:', error);
        return NextResponse.json({ error: 'Unable to retrieve book cover.' }, { status: 502 });
    }
}
