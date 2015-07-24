reset
set terminal png
set output "result.png"
set style data lines
set key right

set ytics nomirror
set y2tics

set xlabel "Training iterations"
set ylabel "Training loss"
set y2label "Test accuracy"
plot "result.train" using 1:3 title "Training loss","result.test" using 1:3 title "Test accuracy" axes x1y2

