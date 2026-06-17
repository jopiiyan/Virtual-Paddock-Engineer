-- Phase 1a — run this once in the Supabase SQL editor.
-- nomic-embed-text produces 768-dim vectors; the column AND the function must both say vector(768).

create extension if not exists vector;

create table documents (
    id bigserial primary key,
    content text,
    metadata jsonb,
    embedding vector(768)
);

create function match_documents (
    query_embedding vector(768),
    match_count int default null,
    filter jsonb default '{}'
) returns table (
    id bigint,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
    return query
    select
        id, content, metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
    from documents
    where metadata @> filter
    order by documents.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- <=> is cosine distance and must match the vector_cosine_ops index opclass.
create index on documents using hnsw (embedding vector_cosine_ops);
