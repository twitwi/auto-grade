
# A dataset of pen trajectories

http://archive.ics.uci.edu/ml/machine-learning-databases/uji-penchars/version2/
http://archive.ics.uci.edu/ml/machine-learning-databases/uji-penchars/version2/ujipenchars2.txt

While writing letters and a few characters, total of 11640 instances.

26 letters + 9 digits + 21 things






## gen dataset (single stroke size)

Make all SVG

    rm -f test-*.svg
    python3 uji2.
    mkdir all-svg
    mv test-* all-svg/

They are not cropped, so there is some work to do...

We recrop at export time

    cd all-svg/
    mkdir ../all-png/
    
    for i in *:* ; do mv $i ${i/:/_colon_} ; done
    
    (for i in *.svg ; do
        i=${i/:/\:}
        echo -b white --export-area-drawing --export-png=../all-png/${i%.svg}.png $i
    done) | inkscape --shell

    #for i in *_colon_* ; do mv $i ${i/_colon_/:} ; done
    cd ../all-png/
    #for i in *_colon_* ; do mv $i ${i/_colon_/:} ; done
    
    # file test-*|awk '{a=$5>a?$5:a ; b=$7>b?$7:b; } END{print(a " " b)}'
    # #==> 76 x 90 max sizes

    mkdir ../all-jpg/
    for i in *.png ; do
        convert $i -gravity center -background white -extent 80x90 ../all-jpg/${i%.png}.jpg
    done
    


~~~
for i in *.png; do ii=$(sed -e 's@^test-@@g' -e 's@-.*@@g' <<<$i); j=${i#test-$ii-}; cl=$(sed -e 's@^\(.\).*@\1@g' <<<$j); mode=$(sed 's@.-\(.*\).png@\1@g' <<<$j);  mkdir -p "$mode/$cl";  mv "$i" "$mode/$cl" ; done 


mv test-*-.-test.png …/
# all in upper case
for d  in *; do mv $d/* ${d^^}/ ; rmdir $d; done
~~~