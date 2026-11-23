-- ============================================================================
-- Project Face <-> Student_APP (Attendance App) Supabase Schema
-- Run this script once in your Supabase SQL Editor.
-- Allows the primary PC (project-face) and remote mobile devices (Student_APP)
-- to sync over any Wi-Fi network via Supabase Cloud without being on the same LAN.
-- ============================================================================

create extension if not exists pgcrypto;

-- 1. Existing face-recognition table (used by vision.py & supabase_sync.py)
create table if not exists public.face_data (
    id_name text primary key,
    name_student text not null default '',
    m text not null default '-',
    array_faceshape jsonb not null default '[]'::jsonb,
    updated_at timestamptz not null default now()
);

alter table public.face_data add column if not exists name_student text not null default '';

-- 2. Normalized student roster synced from primary PC (face_data + face_meta + students.xlsx)
create table if not exists public.attendance_students (
    student_id text primary key,
    full_name text not null,
    classroom text not null default '-',
    leave_status text not null default '',
    is_excused boolean not null default false,
    active boolean not null default true,
    updated_at timestamptz not null default now()
);

alter table public.attendance_students add column if not exists updated_at timestamptz not null default now();

-- 3. Daily attendance records (scans from primary PC + teacher overrides from Student_APP)
create table if not exists public.attendance_records (
    id text primary key,
    student_id text not null references public.attendance_students(student_id) on delete cascade,
    attendance_date date not null,
    status text not null check (status in ('present', 'late', 'absent', 'leave', 'pending', 'unknown')),
    first_scan_at timestamptz null,
    is_finalized boolean not null default false,
    audit_entries jsonb not null default '[]'::jsonb,
    updated_at timestamptz not null default now(),
    unique (student_id, attendance_date)
);

create index if not exists idx_attendance_records_date
    on public.attendance_records (attendance_date);
create index if not exists idx_attendance_records_student_date
    on public.attendance_records (student_id, attendance_date desc);

-- 4. Parent notifications (emitted on attendance check-in or teacher status change)
create table if not exists public.parent_notifications (
    id text primary key,
    student_id text not null,
    student_name text not null,
    title text not null,
    body text not null,
    target_date date not null,
    new_status text not null,
    previous_status text null,
    created_at timestamptz not null default now(),
    is_read boolean not null default false
);

create index if not exists idx_parent_notifications_student_created
    on public.parent_notifications (student_id, created_at desc);

-- 5. Primary PC sync heartbeat
create table if not exists public.sync_status (
    device_id text primary key,
    last_synced_at timestamptz not null default now(),
    total_students integer not null default 0,
    present_students integer not null default 0,
    note text null
);

alter table public.sync_status add column if not exists total_students integer not null default 0;
alter table public.sync_status add column if not exists present_students integer not null default 0;
alter table public.sync_status add column if not exists note text null;

-- 6. Teacher / admin accounts (password hashed; never readable via anon SELECT)
create table if not exists public.teacher_accounts (
    teacher_id text primary key,
    identifier text not null unique,
    password_hash text not null,
    display_name text not null,
    role_label text not null default 'คุณครู',
    active boolean not null default true,
    created_at timestamptz not null default now()
);

-- Default sample/admin account (change password in production)
insert into public.teacher_accounts (teacher_id, identifier, password_hash, display_name, role_label)
values (
    'T-ADMIN-01',
    'admin@khunyuam.ac.th',
    crypt('sample1234', gen_salt('bf')),
    'นายวีระพันธ์ วงศ์สุวรรณ',
    'รองผู้อำนวยการกลุ่มบริหารวิชาการ'
)
on conflict (teacher_id) do nothing;

-- RPC for verifying teacher credentials without exposing password hashes to anon clients
create or replace function public.verify_teacher_login(
    p_identifier text,
    p_password text
)
returns table (
    teacher_id text,
    display_name text,
    role_label text,
    credential_used text
)
language sql
security definer
set search_path = public, extensions
as $$
    select
        t.teacher_id,
        t.display_name,
        t.role_label,
        t.identifier as credential_used
    from public.teacher_accounts t
    where t.active = true
      and lower(trim(t.identifier)) = lower(trim(p_identifier))
      and t.password_hash = crypt(p_password, t.password_hash)
    limit 1;
$$;

revoke all on function public.verify_teacher_login(text, text) from public;
grant execute on function public.verify_teacher_login(text, text) to anon, authenticated;

-- ============================================================================
-- Row-Level Security (RLS)
-- ============================================================================
alter table public.face_data enable row level security;
alter table public.attendance_students enable row level security;
alter table public.attendance_records enable row level security;
alter table public.parent_notifications enable row level security;
alter table public.sync_status enable row level security;
alter table public.teacher_accounts enable row level security;

-- Drop & recreate policies idempotently
drop policy if exists "anon_read_face_data" on public.face_data;
drop policy if exists "anon_all_face_data" on public.face_data;
create policy "anon_all_face_data" on public.face_data
    for all to anon, authenticated using (true) with check (true);

drop policy if exists "anon_read_students" on public.attendance_students;
create policy "anon_read_students" on public.attendance_students
    for select to anon, authenticated using (true);

drop policy if exists "anon_insert_students" on public.attendance_students;
create policy "anon_insert_students" on public.attendance_students
    for insert to anon, authenticated with check (true);

drop policy if exists "anon_update_students" on public.attendance_students;
create policy "anon_update_students" on public.attendance_students
    for update to anon, authenticated using (true) with check (true);

drop policy if exists "anon_read_attendance" on public.attendance_records;
create policy "anon_read_attendance" on public.attendance_records
    for select to anon, authenticated using (true);

drop policy if exists "anon_insert_attendance" on public.attendance_records;
create policy "anon_insert_attendance" on public.attendance_records
    for insert to anon, authenticated with check (true);

drop policy if exists "anon_update_attendance" on public.attendance_records;
create policy "anon_update_attendance" on public.attendance_records
    for update to anon, authenticated using (true) with check (true);

drop policy if exists "anon_read_notifications" on public.parent_notifications;
create policy "anon_read_notifications" on public.parent_notifications
    for select to anon, authenticated using (true);

drop policy if exists "anon_insert_notifications" on public.parent_notifications;
create policy "anon_insert_notifications" on public.parent_notifications
    for insert to anon, authenticated with check (true);

drop policy if exists "anon_update_notifications" on public.parent_notifications;
create policy "anon_update_notifications" on public.parent_notifications
    for update to anon, authenticated using (true) with check (true);

drop policy if exists "anon_read_sync_status" on public.sync_status;
create policy "anon_read_sync_status" on public.sync_status
    for select to anon, authenticated using (true);

drop policy if exists "anon_insert_sync_status" on public.sync_status;
create policy "anon_insert_sync_status" on public.sync_status
    for insert to anon, authenticated with check (true);

drop policy if exists "anon_update_sync_status" on public.sync_status;
create policy "anon_update_sync_status" on public.sync_status
    for update to anon, authenticated using (true) with check (true);

grant usage on schema public to anon, authenticated;
grant select, insert, update, delete on table public.face_data to anon, authenticated;
grant select, insert, update on table
    public.attendance_students,
    public.attendance_records,
    public.parent_notifications,
    public.sync_status
to anon, authenticated;

notify pgrst, 'reload schema';
