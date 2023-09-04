const field = {
    c : ".console",
    exe : ".console.exe"
};



function _translate(code) {
    document.getElementsByTagName("html")[0].lang = code;
    s = document.getElementsByTagName("span");
    for(let i = 0; i < s.length; i++) {
        s[i].innerHTML = strArr.span[i][code];
    }
}

// © 2023. YuiSanae2f