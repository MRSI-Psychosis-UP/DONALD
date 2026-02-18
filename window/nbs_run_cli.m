function nbs_run_cli(varargin)
% Run MATLAB NBS programmatically from exported inputs.
% Example:
% matlab -batch "nbs_run_cli('export_dir','/path/to/matlab_nbs/...','contrast','0 0 0 1')"

parser = inputParser;
parser.addParameter('export_dir', '', @ischar);
parser.addParameter('nbs_path', '', @ischar);
parser.addParameter('contrast', '', @ischar);
parser.addParameter('test', 't', @ischar);
parser.addParameter('size', 'extent', @ischar);
parser.addParameter('thresh', 3.5, @isnumeric);
parser.addParameter('alpha', 0.05, @isnumeric);
parser.addParameter('perms', 1000, @isnumeric);
parser.addParameter('tail', 'both', @ischar);
parser.addParameter('nthreads', 1, @isnumeric);
parser.addParameter('no_precompute', false);
parser.addParameter('output_mat', 'nbs_results.mat', @ischar);
parser.addParameter('show_view', false);
parser.parse(varargin{:});
opts = parser.Results;

if isempty(opts.export_dir)
    error('export_dir is required.');
end
if ~isfolder(opts.export_dir)
    error('export_dir not found: %s', opts.export_dir);
end

if ~isempty(opts.nbs_path)
    if ~isfolder(opts.nbs_path)
        error('NBS path not found: %s', opts.nbs_path);
    end
    addpath(opts.nbs_path);
end

matrices_dir = fullfile(opts.export_dir, 'matrices');
design_path = fullfile(opts.export_dir, 'designMatrix.txt');
labels_path = fullfile(opts.export_dir, 'nodeLabels.txt');
coords_path = fullfile(opts.export_dir, 'COG.txt');

if ~isfolder(matrices_dir)
    error('matrices folder missing: %s', matrices_dir);
end
if ~isfile(design_path)
    error('designMatrix.txt missing: %s', design_path);
end
if ~isfile(labels_path)
    error('nodeLabels.txt missing: %s', labels_path);
end
if ~isfile(coords_path)
    error('COG.txt missing: %s', coords_path);
end

contrast = parse_contrast(opts.contrast);
if isempty(contrast)
    error('contrast is required. Example: ''0 0 0 1'' or ''0 0 0 1;0 0 0 -1''');
end

matrix_files = dir(fullfile(matrices_dir, 'subject*.txt'));
if isempty(matrix_files)
    error('No subject*.txt files found in %s', matrices_dir);
end
first_matrix = fullfile(matrices_dir, matrix_files(1).name);

UI = struct();
UI.method.ui = 'Run NBS';
UI.test.ui = map_test(opts.test);
UI.size.ui = map_size(opts.size);
UI.thresh.ui = num2str(opts.thresh);
UI.perms.ui = num2str(opts.perms);
UI.alpha.ui = num2str(opts.alpha);
UI.contrast.ui = format_contrast(contrast);
UI.design.ui = design_path;
UI.exchange.ui = '';
UI.matrices.ui = first_matrix;
if logical(opts.show_view)
    UI.node_coor.ui = coords_path;
    UI.node_label.ui = labels_path;
else
    % Keep view off in batch mode to avoid graphics-related crashes.
    UI.node_coor.ui = '';
    UI.node_label.ui = '';
end

global nbs
nthreads = round(double(opts.nthreads));
if ~isfinite(nthreads) || nthreads < 1
    nthreads = 1;
end
no_precompute = false;
if islogical(opts.no_precompute)
    no_precompute = opts.no_precompute;
elseif isnumeric(opts.no_precompute)
    no_precompute = opts.no_precompute ~= 0;
elseif ischar(opts.no_precompute) || isstring(opts.no_precompute)
    token = lower(strtrim(char(opts.no_precompute)));
    no_precompute = any(strcmp(token, {'1', 'true', 'yes', 'on'}));
end
nbs.runtime = struct('nthreads', nthreads, 'no_precompute', logical(no_precompute));
NBSrun(UI, struct());
save(fullfile(opts.export_dir, opts.output_mat), 'nbs');

end

function contrast = parse_contrast(raw)
if isempty(raw)
    contrast = [];
    return;
end
rows = strsplit(strtrim(raw), ';');
contrast = [];
for i = 1:numel(rows)
    row = strtrim(rows{i});
    if isempty(row)
        continue;
    end
    parts = regexp(row, '[,\s]+', 'split');
    vals = str2double(parts);
    if any(isnan(vals))
        error('Invalid contrast row: %s', row);
    end
    contrast = [contrast; vals]; %#ok<AGROW>
end
end

function test_ui = map_test(value)
value = lower(strtrim(value));
if strcmp(value, 't')
    test_ui = 't-test';
elseif strcmp(value, 'f')
    test_ui = 'F-test';
else
    test_ui = 't-test';
end
end

function size_ui = map_size(value)
value = lower(strtrim(value));
if strcmp(value, 'intensity')
    size_ui = 'Intensity';
else
    size_ui = 'Extent';
end
end

function expr = format_contrast(contrast)
rows = size(contrast, 1);
parts = cell(rows, 1);
for r = 1:rows
    parts{r} = strtrim(sprintf('%g ', contrast(r, :)));
end
expr = ['[' strjoin(parts, ';') ']'];
end
