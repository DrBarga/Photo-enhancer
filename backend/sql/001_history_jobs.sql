create table if not exists image_history (
  id text primary key,
  created_at timestamptz not null,
  modes jsonb not null default '[]'::jsonb,
  prompt text not null default '',
  total_score integer,
  problem text,
  material text,
  strength text,
  problem_level text,
  result_thumb text,
  payload jsonb not null
);

create index if not exists image_history_created_at_idx
  on image_history (created_at desc);

create index if not exists image_history_problem_idx
  on image_history (problem, strength, material);

create table if not exists image_jobs (
  id text primary key,
  kind text not null,
  status text not null,
  progress integer not null default 0,
  created_at timestamptz not null,
  updated_at timestamptz not null,
  metadata jsonb not null default '{}'::jsonb,
  result jsonb,
  error jsonb
);

create index if not exists image_jobs_status_updated_idx
  on image_jobs (status, updated_at desc);
