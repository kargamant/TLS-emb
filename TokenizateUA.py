#Функция, разбивающая юзер-агент на токены
def TokenizateUA(ua_str):
    is_braces_open = False
    last = 0
    for i in range(len(ua_str)):
        if (ua_str[i]==' ' and i == last): last = i+1
        elif (not is_braces_open) and ua_str[i]==' ':
            yield ua_str[last:i]
            last = i+1
        if ua_str[i] == '(':
            is_braces_open = True
            last = i+1
        elif ua_str[i] == ')' and is_braces_open:
            yield ua_str[last:i]
            last = i+1
            is_braces_open=False
    
    if last < len(ua_str)-1 : yield ua_str[last:]

#Пример работы:
#
#User-agent:
#Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.72 Safari/537.36
#
#Tokens:
#Mozilla/5.0
#Windows NT 10.0; Win64; x64
#AppleWebKit/537.36
#KHTML, like Gecko
#Chrome/89.0.4389.72
#Safari/537.36

