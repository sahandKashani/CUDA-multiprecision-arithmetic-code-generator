for svg in *.svg; do
    svg_filename=$(basename "$svg")
    svg_extension="${svg_filename##*.}"
    png_filename="${svg_filename%.*}"
    echo "$png_filename".pdf

    inkscape -f "$svg_filename" -A "$png_filename".pdf
    inkscape -f "$svg_filename" -e "$png_filename".png
done
