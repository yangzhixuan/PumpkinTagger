function extract_features(file :: String; output :: String = "", brown_features = "")
    fin = open(file)
    lines = map(split, readlines(fin))

    path = Dict{UTF8String, UTF8String}()
    freq = Dict{UTF8String, Int}()
    use_brown = false

    if brown_features != ""
        use_brown = true
        fbrown = open(brown_features)
        while !eof(fbrown)
            str = readline(fbrown)
            line = split(str)
            if length(line) != 3
                msg = @sprintf "invalid line in %s: %s" brown_features str
                warn(msg)
                continue
            end
            path[line[2]] = line[1]
            try
                freq[line[2]] = parseint(line[3])
            catch
                freq[line[2]] = 1
            end
        end
    end

    res = Array(Array{UTF8String}, 0)
    for i in 1:length(lines)
        if length(lines[i]) == 0
            # empty line
            continue
        end

        features = UTF8String[]

        if length(lines[i]) > 1
            push!(features, lines[i][2])
        else
            push!(features, "__TAG_UNKOWN__")
        end

        ##################### words nearby #################### 
        for j in 0:-1:-2
            if i + j < 1 || i + j > length(lines) || length(lines[i+j]) == 0
                break
            end
            f = @sprintf "w[%d]=%s" j lines[i+j][1]
            push!(features, f)
        end

        for j in 1:1:2
            if i + j < 1 || i + j > length(lines) || length(lines[i+j]) == 0
                break
            end
            f = @sprintf "w[%d]=%s" j lines[i+j][1]
            push!(features, f)
        end

        ################# words nearby(bigram) #################### 
        if i > 1 && length(lines[i-1]) > 0
            f = @sprintf "w[%d]|w[%d]=%s|%s" (-1) 0 lines[i-1][1] lines[i][1]
            push!(features, f)
        end

        if i < length(lines) && length(lines[i+1]) > 0
            f = @sprintf "w[%d]|w[%d]=%s|%s" 0 1 lines[i][1] lines[i+1][1]
            push!(features, f)
        end

        # if i > 1 && length(lines[i-1]) > 0 && i < length(lines) && length(lines[i+1]) > 0
        #     f = @sprintf "w[%d]|w[%d]|w[%d]=%s|%s|%s" (-1) 0 1 lines[i-1][1] lines[i][1] lines[i+1][1]
        #     push!(features, f)
        # end


        ################## character features ####################
        for j in 0:1:0
            if i+j < 1 || i+j > length(lines) || length(lines[i+j]) == 0
                continue
            end
            cs = collect(lines[i+j][1])
            for k in max(2, (length(cs) - 2)):length(cs)
                f = @sprintf "w[%d][-%d..-1]=%s" j (length(cs) - k + 1) join(cs[k:end], "")
                push!(features, f)
            end

            for k in 1: min((length(cs)-1), 2)
                f = @sprintf "w[%d][1..%d]=%s" j k join(cs[1:k], "")
                push!(features, f)
            end
        end

        ################# brown features ########################
        if use_brown
            if haskey(path, lines[i][1])
                bits = collect(path[lines[i][1]])
                for j in 1:length(bits)
                    f = @sprintf "bits[1..%d]=%s" j join(bits[1:j], "")
                    push!(features, f)
                end
            end

            if haskey(freq, lines[i][1])
                count = freq[lines[i][1]]
                if count <= 100
                    f = @sprintf "count=%d" count
                else
                    f = @sprintf "count>100"
                end
                push!(features, f)
            end
        end


        if i == 1 || length(lines[i-1]) == 0
            push!(features, "__BOS__")
        end

        if i == length(lines) || length(lines[i+1]) == 0
            push!(features, "__EOS__")
        end


        push!(res, features)
    end

    if output != ""
        try
            str = join(map(x -> join(x, "\t"), res), "\n")
            fout = open(output, "w")
            write(fout, str)
        catch err
            @printf "%s\n" err
        end
    end
    res
end

function read_word2vec_txt(fname :: String)
    lines = readlines(open(fname))
    dim = int(split(lines[1], " ")[2])
    embedding = Dict{String, Array{Float64}}()

    count = 0
    for l in lines[2:end]
        count += 1
        if count % 10000 == 0
            @printf "%d lines are read\n" count
        end

        splitted = split(l)
        a = zeros(dim)
        for (ind, field) in enumerate(splitted[2:end])
            try 
                a[ind] = float(field)
            catch err
                @printf "invalid float: %s\n" field
            end
        end
        embedding[splitted[1]] = a
    end
    embedding
end
