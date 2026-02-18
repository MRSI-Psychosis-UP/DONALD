function nbs_worker_loop(session_dir)
% Persistent MATLAB worker loop for repeated NBS CLI runs.
% Receives commands as JSON files in <session_dir>/commands and writes
% responses to <session_dir>/responses.

if nargin < 1 || isempty(session_dir)
    error('session_dir is required.');
end

session_dir = local_to_char(session_dir);
if ~isfolder(session_dir)
    mkdir(session_dir);
end
commands_dir = fullfile(session_dir, 'commands');
responses_dir = fullfile(session_dir, 'responses');
if ~isfolder(commands_dir)
    mkdir(commands_dir);
end
if ~isfolder(responses_dir)
    mkdir(responses_dir);
end

ready_path = fullfile(session_dir, 'ready.json');
shutdown_path = fullfile(session_dir, 'shutdown.flag');
cleanup_obj = onCleanup(@() local_cleanup_ready(ready_path)); %#ok<NASGU>

ready = struct();
ready.pid = feature('getpid');
ready.started = datestr(now, 30);
local_write_json(ready_path, ready);
fprintf('[NBS-WORKER] Ready. Session: %s\n', session_dir);

while true
    if isfile(shutdown_path)
        fprintf('[NBS-WORKER] Shutdown requested.\n');
        try
            delete(shutdown_path);
        catch
        end
        break;
    end

    files = dir(fullfile(commands_dir, '*.json'));
    if isempty(files)
        pause(0.15);
        continue;
    end

    [~, order] = sort({files.name});
    files = files(order);
    for i = 1:numel(files)
        cmd_path = fullfile(commands_dir, files(i).name);
        [~, stem] = fileparts(files(i).name);
        response_path = fullfile(responses_dir, [stem '.json']);
        if ~isfile(cmd_path)
            continue;
        end

        payload = struct();
        try
            payload = jsondecode(fileread(cmd_path));
        catch ME
            msg = local_error_report(ME);
            local_write_response(response_path, false, msg, stem);
            fprintf('[NBS-WORKER] Failed parsing command %s: %s\n', files(i).name, ME.message);
            try
                delete(cmd_path);
            catch
            end
            continue;
        end

        job_id = stem;
        if isfield(payload, 'job_id') && ~isempty(payload.job_id)
            job_id = local_to_char(payload.job_id);
        end
        if isfield(payload, 'matlab_call')
            matlab_call = local_to_char(payload.matlab_call);
        else
            matlab_call = '';
        end
        if isempty(strtrim(matlab_call))
            local_write_response(response_path, false, 'Missing matlab_call.', job_id);
            fprintf('[NBS-WORKER] Job %s missing matlab_call.\n', job_id);
            try
                delete(cmd_path);
            catch
            end
            continue;
        end

        fprintf('[NBS-WORKER] Job %s started.\n', job_id);
        ok = true;
        err_message = '';
        try
            rehash;
            eval(matlab_call);
        catch ME
            ok = false;
            err_message = local_error_report(ME);
            fprintf('[NBS-WORKER] Job %s failed: %s\n', job_id, ME.message);
        end
        local_write_response(fullfile(responses_dir, [job_id '.json']), ok, err_message, job_id);
        if ok
            fprintf('[NBS-WORKER] Job %s finished.\n', job_id);
        end
        try
            delete(cmd_path);
        catch
        end
    end
end

fprintf('[NBS-WORKER] Exiting.\n');
end


function local_write_response(path, ok, message, job_id)
response = struct();
response.ok = logical(ok);
response.message = local_to_char(message);
response.job_id = local_to_char(job_id);
response.timestamp = datestr(now, 30);
local_write_json(path, response);
end


function local_write_json(path, payload)
tmp_path = sprintf('%s.tmp.%d', path, randi(1e9));
fid = fopen(tmp_path, 'w');
if fid < 0
    error('Could not open temp file for writing: %s', tmp_path);
end
fwrite(fid, jsonencode(payload), 'char');
fclose(fid);
movefile(tmp_path, path, 'f');
end


function text = local_to_char(value)
if isstring(value)
    text = char(value);
elseif ischar(value)
    text = value;
elseif isnumeric(value) || islogical(value)
    text = char(string(value));
else
    text = char(string(value));
end
end


function msg = local_error_report(ME)
try
    msg = getReport(ME, 'extended', 'hyperlinks', 'off');
catch
    msg = local_to_char(ME.message);
end
end


function local_cleanup_ready(ready_path)
if isfile(ready_path)
    try
        delete(ready_path);
    catch
    end
end
end
