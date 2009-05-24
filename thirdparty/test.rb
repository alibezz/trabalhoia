require 'redeneural'
require 'pp'
require 'getoptlong'

def train(net)
  result_ball = [1, 0, 0]
  result_racket = [0, 1, 0]
  result_redbull = [0, 0, 1]
  base_dir = "../images/segmented/"

  results = []

  classes = ["ball", "racket", "redbull"]

  classes.each {|cl|
    File.open(base_dir + cl + "/results") {|f|
      while !f.eof?
        results.push([f.readline.split.map{|it| it.to_f}, eval("result_#{cl}")])
      end
    }
  }

  1000.times {
    results.each { |example, result|
      net.train(example, result)
    }
  }
end

def test(net)
  ref_base_dir = "../images/reference/"

  File.open(ref_base_dir + "/results") {|f|
    puts "--------"
    while !f.eof?
      puts net.eval(f.readline.split.map{|it| it.to_f})
      puts "--------"
    end
  }
end

def read_from_file(file)
  weights = nil
  activation_nodes = nil
  File.open(file, 'r') { |f|
    str = f.readlines.map {|line| line.chomp}.join("*")
    tmp = str.split(/\*___\*/)
    weights = tmp[0].split(/\*---\*/).map {|m| m.split("*") }.map {|n| n.map {|p| p.split.map {|weight| weight.sub(/\*/, "").to_f}}}
    activation_nodes = tmp[1].split("*").map {|m| m.split.map {|act| act.sub(/\*/, "").to_f}}
  }
  [weights, activation_nodes]
end

def write_to_file(file, weights, activation_nodes)
  File.open(file, 'w') { |f|
    f << weights.map {|m| m.map {|n| n.map {|p| p.to_s }.join(' ')}.join("\n")}.join("\n---\n")
    f << "\n___\n"
    f << activation_nodes.map {|m| m.map {|n| n.to_s }.join(' ')}.join("\n")
  }
end

def main
  opts = GetoptLong.new([ '--input-file', '-i', GetoptLong::REQUIRED_ARGUMENT],
                        [ '--output-file', '-o', GetoptLong::REQUIRED_ARGUMENT]
                        )

  opt_file = nil
  ipt_file = nil

  opts.each do |opt, arg|
    case opt
    when '--input-file'
      ipt_file = arg
    when '--output-file'
      opt_file = arg
    end
  end

  net = Ai4r::NeuralNetwork::Backpropagation.new([4, 6, 3])

  if !ipt_file
    train(net)
  else
    net.weights, net.activation_nodes = read_from_file(ipt_file)
  end

  test(net)

  if opt_file
    write_to_file(opt_file, net.weights, net.activation_nodes)
  end
end


main
