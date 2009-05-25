require 'redeneural'
require 'pp'
require 'getoptlong'

# Treina a rede.
# Além da rede a ser trainada, é passado o número de iterações
# como parâmetro. O valor padrão é 1000.
def train(net, iterations=1000)
  # resultados esperados para cada objeto
  result_ball = [1, 0, 0]
  result_racket = [0, 1, 0]
  result_redbull = [0, 0, 1]
  base_dir = "../images/segmented/"

  results = []

  # objetos que temos
  classes = ["ball", "racket", "redbull"]

  # lemos as entradas para realizar o treinamento
  classes.each {|cl|
    File.open(base_dir + cl + "/results") {|f|
      while !f.eof?
        results.push([f.readline.split.map{|it| it.to_f}, eval("result_#{cl}")])
      end
    }
  }

  # aqui a rede é treinada objetivamente
  iterations.times {
    results.each { |example, result|
      net.train(example, result)
    }
  }
end

# Testa a rede.
# Além da rede a ser testada, é passado o arquivo contendo
# as entradas a ser testado.
def test(net, file)
  File.open(file) {|f|
    puts "--------"
    while !f.eof?
      result = net.eval(f.readline.split.map{|it| it.to_f})
      puts format("Bola: %.2f
Raquete: %.2f
Red Bull: %.2f", *result)
      puts "--------"
    end
  }
end

# Lê um arquivo contendo pesos e nós de ativação.
# Retorna um array com os pesos e nós de ativação:
# => [pesos, nos_de_ativacao]
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

# Escreve os pesos e nós de ativação em um arquivo.
def write_to_file(file, weights, activation_nodes)
  File.open(file, 'w') { |f|
    f << weights.map {|m| m.map {|n| n.map {|p| p.to_s }.join(' ')}.join("\n")}.join("\n---\n")
    f << "\n___\n"
    f << activation_nodes.map {|m| m.map {|n| n.to_s }.join(' ')}.join("\n")
  }
end

def main
  # definição das opções de linhas de comando
  opts = GetoptLong.new([ '--input-file', '-i', GetoptLong::REQUIRED_ARGUMENT],
                        [ '--output-file', '-o', GetoptLong::REQUIRED_ARGUMENT],
                        [ '--test-file', '-f', GetoptLong::REQUIRED_ARGUMENT],
                        [ '--iterations', '-n', GetoptLong::REQUIRED_ARGUMENT]
                        )

  opt_file = nil
  ipt_file = nil
  iterations = 1000
  test_file = "../images/reference/results"

  opts.each do |opt, arg|
    case opt
    when '--input-file'
      ipt_file = arg
    when '--output-file'
      opt_file = arg
    when '--test-file'
      test_file = arg
    when '--iterations'
      n = arg.to_i
      if n > 0
        iterations = n
      end
    end
  end

  # cria a rede neural e define o número de neurônios de cada camada
  net = Ai4r::NeuralNetwork::Backpropagation.new([4, 6, 3])

  if !ipt_file
    train(net, iterations)
  else
    net.weights, net.activation_nodes = read_from_file(ipt_file)
  end

  test(net, test_file)

  if opt_file
    write_to_file(opt_file, net.weights, net.activation_nodes)
  end
end

# executa a rotina principal
main
