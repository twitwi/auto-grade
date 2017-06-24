# auto-grade
Study and/or build tools for automatic correction of exams (mcq and more)


### List of projects

#### AMC

- http://auto-multiple-choice.net/
- http://project.auto-multiple-choice.net/projects/auto-multiple-choice/
- https://bitbucket.org/auto-multiple-choice/auto-multiple-choice

Much used, active, in perl+c+glade.

Features:

- from the authoring to the grading,
- latex or simplified source,
- randomization of blocks/questions/answers,
- generate different sheets,
- rule for grading,
- checking and correcting the automatic box-checked process,
- produce reference correction,
- produce corrected individual sheets

Pain points:

- editing (not any latex), sometimes
- getting the same number of pages for each sheet (at least a warning would be good)
- making it compact
- not 100% trust in the interface etc
- cannot install cleanly 2 versions
- adding out-of-amc exercices need some hack (fake question)

Process: general

- data in .sqlite files (mostly one per step)
- layout.sqlite contains the position of most elements (boxes, corners, name field, ...)

Process: generating pdf and finding box positions

- gui entry is `calcule_mep` that calls `"auto-multiple-choice","meptex"` (which runs `AMC-meptex.pl`) and after `detecte_mep`
- it actually relies on latex generating a `DOC-calage.xy` file (which is kind of easy to parse)
- the name of this file is decided when running `AMC-prepare.pl`
- """unit is sp, such that 65536 × 72.27 sp is one inch).""" (in the doc of the automultiplechoice latex package, doc/sty/automultiplechoice.pdf)
- data access in layout.pm
    - e.g. table layout_box, column xmin,xmax,ymin,ymax
    - saved to layout.sqlite
    - it checks for instance that marks are at the same place in all pages 

Things about latex: how it generates the xy file
- using `/usr/share/texmf/tex/latex/AMC/automultiplechoice.sty`
- wrongchoice{inanswer}{content} # inanswer = what to show in answer box for separateanswersheet
- … AMC@box{#1}{}
- … AMC@box{inanswer}{}
- …… AMC@answerBox@{}{#2}{1}{case:\AMCid@name:\the\AMCid@quest,\the\AMCrep@count}
- …… AMC@answerBox@{}{}{1}{case:\AMCid@name:\the\AMCid@quest,\the\AMCrep@count}
- ……… draw the box, and,
- ……… build a commandname based on the `\AMC@shapename` (can plug new ones...)
- ……… compute Z = AMCchoiceLabelFormat{#1} = AMCchoiceLabelFormat{}
- ……… AMC@shape@square{Z}{#2}{#3}{#4}
- ……… AMC@shape@square{Z}{}{1}{case:...}

The scanned box content now ends up in the db, in `capture.sqlite`.

#### Alternative, prototypes etc

----

In Python, single file using opencv:

https://github.com/RanaSamy/Multiple-choice-auto-correction

- uses `cv2.HoughCircles`
- get the extreme coordinates of these
- rectify with `cv2.getRotationMatrix2D` and `cv2.warpAffine`
- crop, blur
- uses 3 different crops (not sur what regions they are)
- then `cv2.adaptiveThreshold`
- then it seems a little ad'hoc

----


#### Helping building AMC questionaires

python3

https://github.com/jarthurgross/amc_question_creator

----

(web)

https://github.com/avitheque/amc-builder

https://amc-builder.avitheque.net/
