function CSR = gen_laplacian(in_name, h, w)
addpath matting/
addpath gaimc/

%disp(['Working on image ' in_name]);
input = im2double(imread(in_name));
input = imresize(input, [h w]);
%size(input)

%close all
%figure; imshow(input);

[h w c] = size(input);

%disp('Compute Laplacian');
A = getLaplacian1(input, zeros(h, w), 1e-7, 1);

%disp('Save to disk');
n = nnz(A);
[Ai, Aj, Aval] = find(A);
CSC = [Ai, Aj, Aval];
%save(['Input_Laplacian_3x3_1e-7_CSC' int2str(i) '.mat'], 'CSC');

[rp ci ai] = sparse_to_csr(A);
Ai = sort(Ai);
Aj = ci;
Aval = ai;
CSR = [Ai, Aj, Aval];
%save(out_name, 'CSR');
