function x = carac(nome)

% Read image

imagen=imread(nome);

try
% Convert to gray scale

imagen=rgb2gray(imagen);

catch ME
    
end


% Remove all object containing fewer than 30 pixels

imagen = bwareaopen(imagen,30);

% Labelize
imagen = bwlabel(imagen);

x = regionprops(imagen, {'Solidity' 'Eccentricity' 'Extent'});
return