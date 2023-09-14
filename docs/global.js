const field = {
    c : ".console",
    exe : ".console.exe"
};

function _translate(code) {
    langCode = code;

    const aEl = document.getElementsByTagName('a');

    document.getElementsByTagName("html")[0].lang = code;
    for (let i = 0; i < strArr.length; i++) {
        document.getElementById(strArr[i].id).innerHTML = strArr[i][code];
    }

    for (let i = 0; i < aEl.length; i++) {
        const currentURL = aEl[i].href;
        const arr = currentURL.split('#'); // 쿼리 문자열 시작 위치 찾
        const pro0 = arr[0];
        url = pro0.split('?')[0] + `?lang=${code}${arr[1] === undefined ? '' : '#' + arr[1]}`;
        aEl[i].href = url;
    }
}

function endl(count = 1) {
    return '\n'.repeat(count);
}

// var langCode;

function _init() {
    // 주어진 URL 문자열
    const urlString = window.location.href.toString();
    const browserLang = (navigator.language || navigator.userLanguage).split("-")[0];

    // "q" 매개변수의 값을 가져오기
    let langCode0 = urlString.split('?')[1];
    let langCode1 = langCode0 === undefined ? browserLang : langCode0.split('=')[1];
    let langCode = langCode1 === undefined ? browserLang : langCode1.split('#')[0];

    langCode = langCode === null ? browserLang : langCode;

    document.write('<p class="indent" style="color:rgb(95, 95, 95); font-size: 10px;">© 2023. YuiSanae2f</p>');
    document.getElementById("_t").innerHTML = `<h2 id="translate"></h2><div class="box indent"><d onclick="_translate('ko')">한국어</d><br><d onclick="_translate('en')">English</d><br></div>`;

    _translate(langCode);
}

// © 2023. YuiSanae2f