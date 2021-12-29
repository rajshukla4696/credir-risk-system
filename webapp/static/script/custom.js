function($) {
    var infoModal = $('#output-modal');
    $('.#submit-btn').on('click', function () {
        $.ajax({
            type: "GET",
            url: '/call_modal',
            dataType: 'json',
            success: function (data) {
                console.log('ajax executed')
                htmlData = data.credit_risk;
                infoModal.find('.modal-body').html(htmlData);
                infoModal.modal('show');
            }
        });

        return false;
    });
}