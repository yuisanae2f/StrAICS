const field = {
    c : ".console",
    exe : ".console.exe"
};

function _translate(code) {
    langCode = code;

    const aEl = document.getElementsByTagName('a');

    for (let i = 0; i < aEl.length; i++) {
        const currentURL = aEl[i].href;
        const queryStringIndex = currentURL.indexOf('?'); // 쿼리 문자열 시작 위치 찾기

        if (queryStringIndex !== -1) {
            const baseURL = currentURL.substring(0, queryStringIndex); // 쿼리 문자열 제외한 부분 추출
            aEl[i].href = `${baseURL}?lang=${langCode}`;

        } else {
            // 쿼리 문자열이 없는 경우, ?lang= 매개변수를 추가합니다.
            aEl[i].href += `?lang=${langCode}`;
        }
    }

    document.getElementsByTagName("html")[0].lang = code;
    for (let i = 0; i < strArr.length; i++) {
        document.getElementById(strArr[i].id).innerHTML = strArr[i][code];
    }
}

function endl(count = 1) {
    return '\n'.repeat(count);
}

// var langCode;

function _init() {
    // 현재 페이지의 URL에서 쿼리 매개변수 가져오기
    const queryString = window.location.search;

    // URLSearchParams 객체를 사용하여 쿼리 매개변수 파싱
    const urlParams = new URLSearchParams(queryString);

    // "q" 매개변수의 값을 가져오기
    langCode = urlParams.get('lang');
    if(langCode === null) langCode = "ko";

    document.write('<p class="indent" style="color:rgb(95, 95, 95); font-size: 10px;">© 2023. YuiSanae2f</p>');
    document.getElementById("_t").innerHTML = `<h2 id="translate"></h2><div class="box indent"><d onclick="_translate('ko')">한국어</d><br><d onclick="_translate('en')">English</d><br></div>`;

    document.write(langCode);
    _translate(langCode);
}

// © 2023. YuiSanae2f