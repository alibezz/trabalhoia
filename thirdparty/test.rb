require 'redeneural'

net = Ai4r::NeuralNetwork::Backpropagation.new([4, 6, 3])

result_ball = [1, 0, 0]
result_racket = [0, 1, 0]
result_redbull = [0, 0, 1]
base_dir = "../images/segmented/"
ref_base_dir = "../images/reference/"

results = []

classes = ["ball", "racket", "redbull"]

classes.each {|cl|
  File.open(base_dir + cl + "/results") {|f|
    while !f.eof?
      results.push([f.readline.split.map{|it| it.to_f}, eval("result_#{cl}")])
    end
  }
}

10000.times {
  results.each { |example, result|
    net.train(example, result)
  }
}


File.open(ref_base_dir + "/results") {|f|
  puts "--------"
  while !f.eof?
    puts net.eval(f.readline.split.map{|it| it.to_f})
    puts "--------"
  end
}

