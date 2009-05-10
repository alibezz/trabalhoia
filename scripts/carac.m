function x = carac(nome)

% Read image

imagen=imread(nome);


% Convert to gray scale

imagen=rgb2gray(imagen);

% Remove all object containing fewer than 30 pixels

imagen = bwareaopen(imagen,30);

% Labelize
imagen = bwlabel(imagen);

x = regionprops(imagen, {'Area' 'BoundingBox' 'Eccentricity' 'Extent'});
return