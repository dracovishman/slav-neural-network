vlib work
vlog "C:/Users/DELL/Desktop/slav/full_adder.v"
vsim -voptargs=+acc full_adder
vcd file output.vcd
vcd add -r /*
add wave *
run 100ns
quit -sim
