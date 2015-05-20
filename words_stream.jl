type WordInstance
    label :: String
    features :: Array{ (String, Float64) }
    sentence_start :: Bool
    sentence_end :: Bool
end

type WordsStream
    buffer :: String
    fp :: IOStream
end

function WordsStream(path :: String)
    WordsStream("", open(path))
end

function Base.start(w :: WordsStream)
    nothing
end

function Base.done(w :: WordsStream, state :: Nothing)
    if w.buffer != "" && w.buffer != "\n"
        return false
    else
        if eof(w.fp)
            close(w.fp)
            return true
        end
        w.buffer = readline(w.fp)
        while w.buffer == "\n" && !eof(w.fp)
            w.buffer = readline(w.fp)
        end
        if w.buffer == "\n"
            close(w.fp)
            return true
        end
        return false
    end
end

function Base.next(w :: WordsStream, state :: Nothing)
    s = split(w.buffer)
    w.buffer = ""
    label = s[1]
    features = (String, Float64)[]
    is_start = false
    is_end = false

    for i in 2:length(s)
        p = split(s[i], ":")
        scale = 1.0
        if length(p) >= 2
            try
                scale = parsefloat(p[2])
            end
        end
        push!(features, (p[1], scale))
        if p[1] == "__BOS__"
            is_start = true
        end
        if p[1] == "__EOS__"
            is_end = true
        end
    end
    return (WordInstance(label, features, is_start, is_end), state)
end
