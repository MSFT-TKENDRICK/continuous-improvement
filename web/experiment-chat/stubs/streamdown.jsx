// Stub: CopilotKit renders assistant markdown with Streamdown, which pulls in KaTeX, Shiki and
// Mermaid (eval, injected <style>, remote assets). Plain react-markdown + GFM renders the same
// subset safely: raw HTML is not rendered and only same-document links are clickable.
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const PLUGINS = [remarkGfm];

function SafeLink({ href, children }) {
  return typeof href === "string" && href.startsWith("#") ? <a href={href}>{children}</a> : <span className="xc-link">{children}</span>;
}

function SafeImage({ alt }) {
  return alt ? <span className="xc-muted">[image: {alt}]</span> : null;
}

const BASE_COMPONENTS = { a: SafeLink, img: SafeImage };

export function Streamdown({ children, className, components }) {
  const text = typeof children === "string" ? children : "";
  return (
    <div className={className}>
      <ReactMarkdown remarkPlugins={PLUGINS} components={{ ...components, ...BASE_COMPONENTS }}>{text}</ReactMarkdown>
    </div>
  );
}

export default Streamdown;
