import type { SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement>;

function Icon({ children, ...props }: IconProps) {
    return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>
            {children}
        </svg>
    );
}

export function SparkleIcon(props: IconProps) {
    return <Icon {...props}><path d="m12 3-1.4 5.6L5 10l5.6 1.4L12 17l1.4-5.6L19 10l-5.6-1.4L12 3Z" /><path d="m19 16-.7 2.3L16 19l2.3.7L19 22l.7-2.3L22 19l-2.3-.7L19 16Z" /></Icon>;
}

export function BookIcon(props: IconProps) {
    return <Icon {...props}><path d="M4.5 5.5A2.5 2.5 0 0 1 7 3h11.5v16H7A2.5 2.5 0 0 0 4.5 21.5v-16Z" /><path d="M4.5 19A2.5 2.5 0 0 1 7 16.5h11.5" /><path d="M8 7h6" /></Icon>;
}

export function BotIcon(props: IconProps) {
    return <Icon {...props}><rect x="3.5" y="6" width="17" height="14" rx="4" /><path d="M12 3v3M9 12h.01M15 12h.01M8.5 16c1 .8 2.2 1.2 3.5 1.2s2.5-.4 3.5-1.2" /></Icon>;
}

export function MoonIcon(props: IconProps) {
    return <Icon {...props}><path d="M20.2 15.2A8.6 8.6 0 0 1 8.8 3.8 8.7 8.7 0 1 0 20.2 15.2Z" /></Icon>;
}

export function SunIcon(props: IconProps) {
    return <Icon {...props}><circle cx="12" cy="12" r="3.5" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></Icon>;
}

export function SendIcon(props: IconProps) {
    return <Icon {...props}><path d="m21 3-7.4 18-3.8-7.8L2 9.4 21 3Z" /><path d="M9.8 13.2 15 8" /></Icon>;
}

export function PlusIcon(props: IconProps) {
    return <Icon {...props}><path d="M12 5v14M5 12h14" /></Icon>;
}

export function ArrowUpRightIcon(props: IconProps) {
    return <Icon {...props}><path d="M7 17 17 7M8 7h9v9" /></Icon>;
}
