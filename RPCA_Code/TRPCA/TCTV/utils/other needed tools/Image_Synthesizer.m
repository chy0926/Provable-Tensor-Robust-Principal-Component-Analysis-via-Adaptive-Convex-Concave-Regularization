function h = Image_Synthesizer(Image_Original, Position_Window, Imresize_Ratio, Location_Image_Resized, Color_Window, LineStyle_Window)

    % 获取图像尺寸（正确处理彩色和灰度图像）
    [m_Image_Original, n_Image_Original, num_channels] = size(Image_Original);
    
    % 裁剪局部区域
    Image_Local = imcrop(Image_Original, Position_Window);
    
    % 放大图像
    Image_Resized = imresize(Image_Local, Imresize_Ratio);
    
    % 获取放大后图像的尺寸
    [m_Image_Resized, n_Image_Resized, ~] = size(Image_Resized);
    
    % 显示原始图像
    imshow(Image_Original);
    hold on;
    
    % 在原始图像上绘制矩形框标记放大区域
    rectangle('Position', Position_Window, 'EdgeColor', Color_Window, ...
              'LineWidth', 2, 'LineStyle', LineStyle_Window);
    
    h = 1;
    
    % 修正坐标计算：MATLAB中x是水平(宽度)，y是垂直(高度)
    switch Location_Image_Resized 
        case 'LeftTop'  
            x = 3;
            y = 3;
        case 'RightTop'
            x = n_Image_Original - n_Image_Resized - 1;  % 修正：用宽度计算x
            y = 3;                                      % 修正：用高度计算y
        case 'LeftBottom'
            x = 3;
            y = m_Image_Original - m_Image_Resized - 1;  % 修正：用高度计算y
        case 'RightBottom'
            x = n_Image_Original - n_Image_Resized - 1;  % 修正：用宽度计算x
            y = m_Image_Original - m_Image_Resized - 1;  % 修正：用高度计算y
        otherwise
            h = 0;
            disp('Please check your inputs carefully when you are using function ''Image_Synthesizer''.')
            close all;
            return;
    end
    
    % 显示放大图像（修正坐标数据）
    imshow(Image_Resized, 'XData', [x, x + n_Image_Resized - 1], ...
                         'YData', [y, y + m_Image_Resized - 1]);
    
    % 在放大图像周围绘制边框
    Position_Image = [x - 1, y - 1, n_Image_Resized + 1, m_Image_Resized + 1];
    rectangle('Position', Position_Image, 'EdgeColor', Color_Window, ...
              'LineWidth', 2, 'LineStyle', LineStyle_Window);
end